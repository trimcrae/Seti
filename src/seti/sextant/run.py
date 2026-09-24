"""SEXTANT stages: probe, acquire+fit+screen per shard, assess.

The search, in one paragraph.  For every numbered asteroid in
``gaiafpr.sso_observation`` with enough transits, take the observed-minus-
predicted along-scan residual against a full-force prediction from JPL's orbit
(``seti.sextant.ephem``), marginalise the six orbit-state partials so that
nothing an orbit fit could have absorbed is counted, and fit the exact
linearised response to a transverse ``A2`` (and ``A1``/``A3``) under the
archive's own two-part error model.  Gate the result on LOOM's radiation-
momentum ceiling --- a body of that size cannot be pushed harder than sunlight
allows --- and decide on the *set* of objects above it with LOOM's population
statistics against matched draws of this same screened sample.  The positive
controls are the objects whose Yarkovsky drift JPL has already measured: the
Gaia-only fit against a gravity-only prediction must return JPL's ``A2``, sign
and magnitude, or nothing else in the output is believed.

Stages (``python -m seti.cli sextant --stage ... --shard i/n``):

``probe``
    Runner-only, once per run.  Pulls the perturber grids and the SBDB
    catalogue, then on a sample of objects measures (a) the disagreement between
    the local propagator and JPL Horizons, which decides the bulk ephemeris
    route, (b) the archive's astrometric conventions on data (which corrections
    the published positions already contain), and (c) the end-to-end ``A2``
    from both routes.  Writes ``probe_ephemeris.json`` and ``conventions.json``.

``fit`` (per shard; ``acquire`` and ``screen`` are its parts)
    Pulls this shard's objects from Gaia in chunks, propagates every chunk as
    one batch, fits every object, screens every record, and checkpoints after
    each chunk.  Writes ``fits/shard_<i>_of_<n>.csv.gz`` (every object, one
    row) and ``fits/shard_<i>_of_<n>.json`` (the shard's timing offset, route,
    funnel and the full model comparison for anything above ``ordinary``).

``assess``
    Offline.  Gathers every shard, scores the controls, characterises the
    ``A2`` distribution against the Yarkovsky expectation, lists the ceiling
    exceedances with their vetting, runs the population tests, and writes
    ``summary.json`` / ``assessment.json`` / ``controls.json`` /
    ``exceedances.json``.

Nothing here decides anything a test cannot exercise offline: the batch
propagation, the fit, the screen and the assessment are pure given their
inputs, and ``tests/test_sextant_run.py`` drives them on synthetic sky with a
known injected ``A2``.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..loom import nongrav as NG
from ..loom.replication import replication_tests
from . import controls as C
from . import ephem as E
from . import residuals as R
from .controls import PUBLISHED_YARKOVSKY, score_control, summarise_controls
from .screen import BinaryCatalogue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "release": "gaiafpr",
    "objects_per_chunk": 250,
    # An archive outage must stop the shard rather than spend its whole clock
    # recording "no observations" for objects Gaia never answered about.
    "max_consecutive_failed_chunks": 4,
    "failed_chunk_retry_sleep_s": 60.0,
    "max_objects": 0,                    # 0 = every object in the release
    "ephemeris_route": "auto",           # auto | integrator | horizons
    "horizons_min_interval_s": 1.0,
    "probe_sample": 60,
    "probe_conventions_objects": 8,
    "max_integrator_disagreement_mas": 0.5,
    "min_observations": 40,
    "min_transits": 8,
    "min_arc_days": 365.0,
    "min_delta_chi2": 16.0,
    "min_snr_detection": 3.0,
    "min_force_snr": 5.0,
    "max_absorbed_fraction": 0.995,
    "max_excess_scatter": 2.0,
    "epsilon_realistic": NG.EPSILON_REALISTIC,
    "epsilon_hard": NG.EPSILON_HARD,
    "epsilon_median_measured": 0.074,    # LOOM, 589 SBDB asteroids, 2026-07-30
    "population_n_null": 2000,
    "min_parent_for_population": 200,
    "min_anomalies_for_population": 5,
    "control_min_jpl_snr": 3.0,
    "conventions": {"epoch": "epoch:TCB:2455197.5", "apply_light_time": True,
                    "apply_stellar_aberration": False,
                    "apply_solar_deflection": False,
                    "scan_pa_north_to_east": True},
    "seed": 20260921,
}


def load_config(cfg=None) -> dict:
    """Defaults, overlaid by ``config/sextant.yaml`` (``sextant:`` block)."""
    conf = json.loads(json.dumps(DEFAULT_CONFIG))
    block = None
    if isinstance(cfg, dict):
        block = cfg.get("sextant")
    if block is None:
        p = _root() / "config" / "sextant.yaml"
        if p.exists():
            try:
                import yaml

                block = (yaml.safe_load(p.read_text()) or {}).get("sextant")
            except Exception:                                  # noqa: BLE001
                block = None
    if isinstance(block, dict):
        for k, v in block.items():
            if isinstance(v, dict) and isinstance(conf.get(k), dict):
                conf[k].update(v)
            else:
                conf[k] = v
    return conf


def _root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fin(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


@dataclass
class Paths:
    root: Path
    results: Path
    work: Path

    @classmethod
    def make(cls, out_dir=None, work_dir=None) -> Paths:
        root = _root()
        res = Path(out_dir) if out_dir else root / "results" / "sextant"
        work = Path(work_dir) if work_dir else root / "work" / "sextant"
        res.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        return cls(root=root, results=res, work=work)


def conventions_from(conf_block: dict) -> R.Conventions:
    col, scale, zero = str(conf_block["epoch"]).split(":")
    return R.Conventions(
        epoch=R.EpochConvention(col, scale, float(zero)),
        apply_light_time=bool(conf_block.get("apply_light_time", True)),
        apply_stellar_aberration=bool(conf_block.get("apply_stellar_aberration", False)),
        apply_solar_deflection=bool(conf_block.get("apply_solar_deflection", False)),
        scan_pa_north_to_east=bool(conf_block.get("scan_pa_north_to_east", True)),
        resolved_by=str(conf_block.get("resolved_by", "CONFIG_DEFAULT")),
        resolution_margin=_f(conf_block.get("resolution_margin")),
        notes=str(conf_block.get("notes", "")))


def load_conventions(paths: Paths, conf: dict) -> R.Conventions:
    """The probe's measured conventions if it ran, else the config default."""
    p = paths.results / "conventions.json"
    if p.exists():
        try:
            rec = json.loads(p.read_text())
            if rec.get("verdict") == "RESOLVED" and rec.get("conventions"):
                return conventions_from(rec["conventions"])
        except Exception:                                      # noqa: BLE001
            pass
    return conventions_from(conf["conventions"])


# ---------------------------------------------------------------------------
# Orbit-source provenance for each route
# ---------------------------------------------------------------------------
def orbit_source_for(route: str, sbdb_row: dict | None) -> R.OrbitSource:
    """The provenance record the residual chain requires, per route and object."""
    ng = bool(sbdb_row and sbdb_row.get("nongrav_fitted"))
    if route == "horizons":
        return R.OrbitSource(
            name="jpl_horizons", provider="JPL Horizons (VECTORS, CENTER=500@0)",
            dynamical_model="jpl_nbody_de44x_plus_perturbers",
            solution_reference="JPL SBDB small-body solution, current",
            gaia_sso_astrometry_in_fit=True, nongrav_parameters_fitted=ng,
            nongrav_in_prediction=ng,
            notes="Horizons integrates the current JPL solution with its A1/A2/A3 if any")
    if route == "horizons_pinned_gravity_only":
        return R.OrbitSource(
            name="jpl_horizons_pinned_gravity_only",
            provider="JPL Horizons state at mid-arc, propagated gravity-only here",
            dynamical_model="assist_nbody",
            solution_reference="JPL SBDB small-body solution, current",
            gaia_sso_astrometry_in_fit=True, nongrav_parameters_fitted=ng,
            nongrav_in_prediction=False,
            notes="positive-control route: the fitted A2 is re-inserted into the residual")
    if route == "integrator":
        return R.OrbitSource(
            name="jpl_sbdb_elements_nbody",
            provider="JPL SBDB osculating elements (full-prec), seti.sextant.ephem",
            dynamical_model="assist_nbody",
            solution_reference="JPL SBDB small-body solution, current",
            gaia_sso_astrometry_in_fit=True, nongrav_parameters_fitted=ng,
            nongrav_in_prediction=False,
            notes=("Sun+planets+Moon+Pluto+SB441-N16, Schwarzschild GR; gravity-only "
                   "so a fitted A2 stays in the residual"))
    raise ValueError(f"unknown route {route!r}")


# ---------------------------------------------------------------------------
# Observations -> per-object column dicts
# ---------------------------------------------------------------------------
def group_observations(rows: list[dict]) -> dict[int, dict]:
    """Rows from the Gaia pull, grouped by ``number_mp`` into column dicts."""
    by: dict[int, list[dict]] = {}
    for r in rows:
        n = r.get("number_mp")
        if n is None:
            continue
        by.setdefault(int(n), []).append(r)
    out = {}
    for n, rs in by.items():
        cols = R.as_columns(rs)
        rej = cols.get("is_rejected")
        if rej is not None:
            arr = np.asarray(rej)
            # Anything that is not already numeric or boolean goes through
            # `_truthy`, which decides a string by CONTENT.  `dtype == object`
            # alone was not enough: a column of Python strings becomes a numpy
            # '<U5' array, not an object array, and `'false'.astype(float)`
            # raises rather than mis-reading --- while `bool('false')` is True,
            # which would drop every FPR observation while looking like a
            # working quality cut.  The probe listed the column's type as
            # UNVERIFIED for exactly this reason, so the code refuses to depend
            # on it.
            if arr.dtype.kind in ("O", "U", "S", "V"):
                from .acquire import _truthy

                cols["is_rejected"] = np.array([_truthy(x) for x in arr])
            else:
                cols["is_rejected"] = np.where(np.isfinite(arr.astype(float)),
                                               arr.astype(float) != 0.0, False)
        out[n] = cols
    return out


def rows_to_frame(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def frame_to_rows(df) -> list[dict]:
    """Rows out of a frame, with every flavour of missing value as ``None``.

    ``isinstance(v, float)`` was enough while a missing string sat in an object
    column as ``float('nan')``.  It is not enough in general: a nullable or
    arrow-backed column yields ``pd.NA``, and a datetime column ``pd.NaT``,
    neither of which is a float --- so they would travel downstream as objects
    that are neither a value nor a null, and ``float(pd.NA)`` raises rather
    than returning NaN.  ``pd.isna`` answers for all of them, and is applied
    only to scalars because on a list or array it returns an array.
    """
    import pandas as pd

    recs = df.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, float):
                if not math.isfinite(v):
                    r[k] = None
            elif v is not None and np.ndim(v) == 0 and pd.isna(v):
                r[k] = None
    return recs


# ---------------------------------------------------------------------------
# Ephemeris bundles
# ---------------------------------------------------------------------------
@dataclass
class EphemBundle:
    """What one object needs from the ephemeris: an aligned state and a reference."""

    number: int
    route: str
    target_state: object
    state0_helio: np.ndarray
    jd0: float
    jd_obs: np.ndarray
    notes: list[str] = field(default_factory=list)


def observation_epochs(cols: dict, conv: R.Conventions) -> np.ndarray:
    n = int(np.asarray(cols["ra"]).size)
    return R.epoch_to_jd_tdb(R._col(cols, conv.epoch.column, n), conv.epoch)


def integrator_bundles(objects: dict[int, dict], sbdb: dict[int, dict],
                       pert: E.PerturberSet, conv: R.Conventions,
                       *, relativity: bool = True, on_group=None,
                       start_states: dict[int, tuple[float, np.ndarray]] | None = None
                       ) -> tuple[dict[int, EphemBundle], dict[int, str]]:
    """Propagate every object of a chunk as one batch per (step, epoch) group.

    ``start_states`` overrides the SBDB elements with an explicit barycentric
    state per object (``number -> (jd, state6)``) --- the pinned control route.
    Returns the bundles and, separately, the reason for every object that could
    not be propagated (no SBDB orbit, hyperbolic, NaN elements).
    """
    prop = E.NBodyPropagator(pert, relativity=relativity)
    skipped: dict[int, str] = {}
    groups: dict[tuple[float, float], list[int]] = {}
    starts: dict[int, tuple[float, np.ndarray]] = {}
    for n in objects:
        if start_states and n in start_states:
            jd_e, st = start_states[n]
            starts[n] = (float(jd_e), np.asarray(st, dtype=float).reshape(6))
            row = sbdb.get(n) or {}
            a, e = _f(row.get("a")), _f(row.get("e"))
        else:
            row = sbdb.get(n)
            if not row:
                skipped[n] = "no_sbdb_orbit"
                continue
            a, e = _f(row["a"]), _f(row["e"])
            vals = [a, e, row["i"], row["node"], row["argperi"], row["ma"], row["epoch_jd"]]
            if not all(math.isfinite(_f(v)) for v in vals) or e >= 0.999 or a <= 0:
                skipped[n] = "elements_unusable"
                continue
            if not pert.covers(float(row["epoch_jd"]), float(row["epoch_jd"])):
                # Never start from an epoch the perturbers do not reach: the
                # state there would be built on an extrapolated Sun.
                skipped[n] = (f"osculation_epoch_{float(row['epoch_jd']):.1f}"
                              "_outside_perturber_grid")
                continue
            st_h = E.elements_to_heliocentric_state(
                a, e, row["i"], row["node"], row["argperi"], row["ma"])[0]
            sun = pert.sun_state(np.array([row["epoch_jd"]]))[0]
            starts[n] = (float(row["epoch_jd"]), st_h + sun)
        if not (math.isfinite(a) and math.isfinite(e)):
            a, e = 2.5, 0.2
        h = float(E.step_for(a, e))
        groups.setdefault((h, starts[n][0]), []).append(n)

    bundles: dict[int, EphemBundle] = {}
    for (h, jd_e), members in groups.items():
        st0 = np.array([starts[n][1] for n in members])
        ev_obj, ev_jd, obs_slices, jd0s = [], [], {}, {}
        cursor = 0
        for k, n in enumerate(members):
            jd = observation_epochs(objects[n], conv)
            good = np.isfinite(jd)
            jd_use = jd[good]
            jd0 = float(np.median(jd_use)) if jd_use.size else float("nan")
            jd0s[n] = jd0
            ev_obj.extend([k] * (jd_use.size + 1))
            ev_jd.extend(list(jd_use) + [jd0])
            obs_slices[n] = (cursor, cursor + jd_use.size, good)
            cursor += jd_use.size + 1
        if not ev_jd:
            continue
        ev_jd = np.array(ev_jd, dtype=float)
        ev_obj = np.array(ev_obj, dtype=int)
        fin = np.isfinite(ev_jd)
        t0 = time.time()
        r, v, a_ = prop.propagate(st0, jd_e, E.EvalRequest(ev_obj[fin], ev_jd[fin]), h)
        r_full = np.full((ev_jd.size, 3), np.nan)
        v_full = np.full((ev_jd.size, 3), np.nan)
        a_full = np.full((ev_jd.size, 3), np.nan)
        r_full[fin], v_full[fin], a_full[fin] = r, v, a_
        if on_group is not None:
            on_group({"step_days": h, "epoch_jd": jd_e, "n_objects": len(members),
                      "n_epochs": int(fin.sum()), "seconds": time.time() - t0})
        for n in members:
            lo, hi, good = obs_slices[n]
            jd_all = observation_epochs(objects[n], conv)
            N = jd_all.size
            rr = np.full((N, 3), np.nan)
            vv = np.full((N, 3), np.nan)
            aa = np.full((N, 3), np.nan)
            rr[good], vv[good], aa[good] = r_full[lo:hi], v_full[lo:hi], a_full[lo:hi]
            sun = pert.sun_state(np.where(good, jd_all, jd0s[n]))
            eph = E.AlignedEphemeris(jd_all, rr, vv, aa, sun_r=sun[:, :3])
            st_ref = np.concatenate([r_full[hi], v_full[hi]])
            sun0 = pert.sun_state(np.array([jd0s[n]]))[0]
            bundles[n] = EphemBundle(number=n, route="integrator", target_state=eph,
                                     state0_helio=st_ref - sun0, jd0=jd0s[n],
                                     jd_obs=jd_all)
    return bundles, skipped


def horizons_bundle(number: int, cols: dict, client: E.HorizonsClient,
                    pert: E.PerturberSet, conv: R.Conventions) -> EphemBundle:
    """One Horizons call per object: states at the transit epochs, expanded."""
    jd = observation_epochs(cols, conv)
    n = jd.size
    tid = R._col(cols, "transit_id", n)
    if not np.any(np.isfinite(tid)):
        tid = np.arange(n, dtype=float)
    good = np.isfinite(jd)
    uniq, first = E.transit_reference_epochs(np.where(good, jd, np.nan), tid)
    usable = np.isfinite(first)
    jd0 = float(np.median(jd[good])) if good.any() else float("nan")
    ask = np.concatenate([first[usable], [jd0]])
    _, st = client.vectors_at(f"{int(number)};", ask)
    st_tr = np.full((uniq.size, 6), np.nan)
    st_tr[usable] = st[:-1]
    st0 = st[-1]
    # Transits with no usable epoch get NaN states, which propagate to NaN residuals.
    eph = E.expand_transit_states(np.where(good, jd, first[np.unique(tid, return_inverse=True)[1]]),
                                  tid, np.where(usable, first, jd0), st_tr,
                                  sun_state_fn=pert.sun_state)
    sun0 = pert.sun_state(np.array([jd0]))[0]
    return EphemBundle(number=number, route="horizons", target_state=eph,
                       state0_helio=st0 - sun0, jd0=jd0, jd_obs=jd)


def pinned_bundle(number: int, cols: dict, client: E.HorizonsClient,
                  pert: E.PerturberSet, conv: R.Conventions, sbdb: dict
                  ) -> EphemBundle:
    """The control route: Horizons state at mid-arc, propagated gravity-only."""
    jd = observation_epochs(cols, conv)
    good = np.isfinite(jd)
    jd0 = float(np.median(jd[good]))
    _, st = client.vectors_at(f"{int(number)};", np.array([jd0]))
    bundles, skipped = integrator_bundles({number: cols}, sbdb, pert, conv,
                                          start_states={number: (jd0, st[0])})
    if number not in bundles:
        raise E.EphemerisError(f"pinned propagation failed: {skipped.get(number)}")
    b = bundles[number]
    b.route = "horizons_pinned_gravity_only"
    return b


# ---------------------------------------------------------------------------
# The per-object fit
# ---------------------------------------------------------------------------
def _joint_fit(series: R.ResidualSeries, columns: dict[str, np.ndarray],
               nuisance: np.ndarray) -> dict:
    """Joint GLS of several signal columns with the nuisance marginalised."""
    m = series.usable()
    names = [k for k, c in columns.items() if np.all(np.isfinite(np.asarray(c)[m]))]
    out: dict = {"names": names}
    if len(names) < 2 or m.sum() < 30 or nuisance.shape[0] != m.sum():
        out["verdict"] = "NOT_EVALUABLE"
        return out
    y = series.al_mas[m]
    sig_r = series.sigma_al_random[m]
    sig_s = np.where(np.isfinite(series.sigma_al_systematic[m]),
                     series.sigma_al_systematic[m], 0.0)
    X = np.column_stack([np.asarray(columns[k], dtype=float)[m] for k in names])
    A = np.column_stack([nuisance, X]) if nuisance.size else X
    scales = np.linalg.norm(A, axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    A_s = A / scales
    blocks = series.transit_id[m]
    try:
        ata, aty, yty = R._gls_normal_equations(A_s, y, sig_r, sig_s, blocks)
        p = np.linalg.solve(ata, aty)
        cov = np.linalg.pinv(ata)
    except np.linalg.LinAlgError:
        out["verdict"] = "SINGULAR"
        return out
    chi2 = float(yty - p @ aty)
    dof = int(m.sum() - A_s.shape[1])
    scale = max(math.sqrt(chi2 / dof) if dof > 0 and chi2 > 0 else 1.0, 1.0)
    k0 = nuisance.shape[1] if nuisance.size else 0
    for j, nm in enumerate(names):
        jj = k0 + j
        out[nm] = float(p[jj]) / scales[jj]
        out[f"{nm}_err"] = math.sqrt(max(cov[jj, jj], 0.0)) / scales[jj] * scale
    out["chi2_reduced"] = chi2 / dof if dof > 0 else float("nan")
    out["verdict"] = "OK"
    return out


def fit_object(number: int, cols: dict, bundle: EphemBundle, sbdb_row: dict | None,
               pert: E.PerturberSet, conv: R.Conventions, conf: dict,
               denomination: str | None = None) -> tuple[dict, R.ResidualSeries | None]:
    """Residuals, the six-model comparison, A1/A2/A3, and the record for one object."""
    src = orbit_source_for(bundle.route, sbdb_row)
    rec: dict = {"number_mp": int(number), "denomination": denomination,
                 "route": bundle.route, "n_obs": int(np.asarray(cols["ra"]).size)}
    try:
        series = R.compute_residuals(
            cols, bundle.target_state, src, conv=conv, state0=bundle.state0_helio,
            jd0=bundle.jd0, sun_state=pert.sun_state, allow_partial=True,
            key=str(number))
    except (R.CircularOrbitSourceError, R.UnknownProvenanceError, KeyError,
            ValueError) as exc:
        rec["verdict"] = "RESIDUALS_FAILED"
        rec["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        return rec, None
    s = series.summary()
    rec.update({"n_usable": s["n_usable"], "n_rejected": s["n_rejected"],
                "n_transits": s["n_transits"], "arc_days": s["arc_days"],
                "median_sigma_al_mas": s["median_sigma_al_mas"],
                "rms_al_mas": s["rms_al_mas"], "rms_ac_mas": s["rms_ac_mas"],
                "independence": series.independence})
    m = series.usable()
    if m.any():
        t = series.jd_tdb[m]
        rec["mjd_min"] = float(t.min() - 2400000.5)
        rec["mjd_max"] = float(t.max() - 2400000.5)
        rec["median_r_au"] = float(np.nanmedian(series.r_au[m]))
        rec["median_delta_au"] = float(np.nanmedian(series.delta_au[m]))
    part = R.scan_axis_partition(series)
    rec["chi_al"] = _f(part.get("chi_al"))
    rec["chi_ac"] = _f(part.get("chi_ac"))
    rec["sigma_ratio_ac_over_al"] = _f(part.get("sigma_ratio_ac_over_al"))
    rec["median_track_scan_projection"] = _f(part.get("median_track_scan_projection"))
    if rec["n_transits"] < int(conf["min_transits"]):
        rec["verdict"] = "TOO_FEW_TRANSITS"
        rec["reason"] = f"{rec['n_transits']}_transits_below_{conf['min_transits']}"
        return rec, series
    mc = R.model_comparison(series, min_observations=int(conf["min_observations"]),
                            min_arc_days=float(conf["min_arc_days"]),
                            min_delta_chi2=float(conf["min_delta_chi2"]))
    rec["model_verdict"] = mc.get("verdict")
    rec["best_model"] = mc.get("best_model")
    rec["best_force_model"] = mc.get("best_force_model")
    rec["best_geometric_model"] = mc.get("best_geometric_model")
    rec["family_margin"] = _f(mc.get("family_margin"))
    rec["delta_chi2_force"] = _f(mc.get("delta_chi2_force"))
    rec["delta_chi2_geometric"] = _f(mc.get("delta_chi2_geometric"))
    rec["law_verdict"] = mc.get("law_verdict")
    rec["best_law"] = mc.get("best_law")
    rec["law_margin"] = _f(mc.get("law_margin"))
    rec["nuisance_incomplete"] = bool(mc.get("nuisance_incomplete", False))
    fits = mc.get("fits") or {}
    rad = fits.get("force:radiation") or {}
    rec["a2"] = _f(rad.get("amplitude"))
    rec["a2_err"] = _f(rad.get("amplitude_err_used", rad.get("amplitude_err")))
    rec["a2_err_pessimistic"] = _f(rad.get("amplitude_err_pessimistic"))
    rec["a2_snr"] = _f(rad.get("snr"))
    rec["a2_absorbed_fraction"] = _f(rad.get("absorbed_fraction"))
    rec["a2_chi2_reduced"] = _f(rad.get("chi2_reduced"))
    rec["a2_delta_chi2"] = _f(rad.get("delta_chi2"))
    rec["a2_fit_ok"] = bool(rad.get("ok", False))
    rec["a2_fit_reason"] = rad.get("reason") or ""
    tim = fits.get("geometry:timing") or {}
    rec["timing_dt_s"] = _f(tim.get("dt_seconds"))
    rec["timing_delta_chi2"] = _f(tim.get("delta_chi2"))
    sub = fits.get("force:sublimation") or {}
    con = fits.get("force:constant") or {}
    rec["delta_chi2_sublimation"] = _f(sub.get("delta_chi2"))
    rec["delta_chi2_constant"] = _f(con.get("delta_chi2"))
    rec["excess_scatter"] = (math.sqrt(rec["a2_chi2_reduced"])
                             if math.isfinite(rec["a2_chi2_reduced"]) else float("nan"))
    # A1 and A3 on the same footing, singly and jointly with A2.
    extra: dict[str, np.ndarray] = {}
    nuis = R.nuisance_design(series, m)
    if rad.get("ok") and series.signal_columns.get("radiation") is not None:
        try:
            u_pred = None
            for direction, nm in (("radial", "a1"), ("normal", "a3")):
                resp = R.variational_response(series.state0, series.jd0, series.jd_tdb,
                                              law="radiation", direction=direction)
                if u_pred is None:
                    # Reconstruct the predicted directions from the observer and
                    # the target via the ephemeris, exactly as ingest did.
                    st = np.atleast_2d(bundle.target_state(series.jd_tdb))
                    Rg = np.column_stack([R._col(cols, c, series.n)
                                          for c in ("x_gaia", "y_gaia", "z_gaia")])
                    rho = st[:, :3] - Rg
                    u_pred = rho / np.linalg.norm(rho, axis=1)[:, None]
                    pa = R._col(cols, "position_angle_scan", series.n)
                    e_al, _ = R.scan_basis(pa, north_to_east=conv.scan_pa_north_to_east)
                resp_en = R.project_to_tangent(resp, u_pred)
                with np.errstate(divide="ignore", invalid="ignore"):
                    col = ((resp_en[:, 0] * e_al[:, 0] + resp_en[:, 1] * e_al[:, 1])
                           / series.delta_au * R.MAS_PER_RAD)
                extra[nm] = col
                f = R.fit_model(series, col, name=f"force:{nm}", units="au/day^2",
                                nuisance=nuis,
                                min_observations=int(conf["min_observations"]),
                                min_arc_days=float(conf["min_arc_days"]))
                rec[nm] = _f(f.amplitude)
                rec[f"{nm}_err"] = _f(f.detail.get("amplitude_err_used", f.amplitude_err))
                rec[f"{nm}_snr"] = _f(f.snr)
                rec[f"{nm}_absorbed_fraction"] = _f(f.absorbed_fraction)
                fits[f.name] = f.as_dict()
            joint = _joint_fit(series, {"a1": extra["a1"],
                                        "a2": series.signal_columns["radiation"],
                                        "a3": extra["a3"]}, nuis)
            rec["a2_joint"] = _f(joint.get("a2"))
            rec["a2_joint_err"] = _f(joint.get("a2_err"))
            rec["a1_joint"] = _f(joint.get("a1"))
            rec["a3_joint"] = _f(joint.get("a3"))
            rec["joint_verdict"] = joint.get("verdict")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            rec["a1a3_note"] = f"{type(exc).__name__}: {exc}"[:120]
    rec["_fits"] = fits
    rec["_law_separability"] = mc.get("law_separability")
    rec["verdict"] = "FITTED" if rec["a2_fit_ok"] else "FIT_REFUSED"
    if not rec["a2_fit_ok"]:
        rec["reason"] = rec["a2_fit_reason"]
    return rec, series


# ---------------------------------------------------------------------------
# The screen (pure)
# ---------------------------------------------------------------------------
TIERS = ("untestable", "ordinary", "watch", "interest", "candidate")


def annotate_orbit(rec: dict, sbdb_row: dict | None, conf: dict) -> dict:
    """Attach H, elements, the JPL non-grav solution and the ceilings."""
    row = sbdb_row or {}
    for k in ("h", "g", "albedo", "diameter_km", "a", "e", "i", "node", "argperi",
              "q", "n_obs_used", "data_arc", "rms", "condition_code", "n_opp"):
        rec[k] = _f(row.get(k))
    rec["orbit_class"] = row.get("orbit_class")
    rec["neo"] = row.get("neo")
    rec["jpl_a1"] = _f(row.get("a1"))
    rec["jpl_a2"] = _f(row.get("a2"))
    rec["jpl_a3"] = _f(row.get("a3"))
    rec["jpl_a2_sigma"] = _f(row.get("a2_sigma"))
    rec["jpl_nongrav_fitted"] = bool(row.get("nongrav_fitted", False))
    rec["jpl_a2_snr"] = (abs(rec["jpl_a2"]) / rec["jpl_a2_sigma"]
                         if math.isfinite(rec["jpl_a2"]) and math.isfinite(rec["jpl_a2_sigma"])
                         and rec["jpl_a2_sigma"] > 0 else float("nan"))
    rec["is_control"] = bool(math.isfinite(rec["jpl_a2_snr"])
                             and rec["jpl_a2_snr"] >= float(conf["control_min_jpl_snr"]))
    rec["published_control"] = PUBLISHED_YARKOVSKY.get(int(rec["number_mp"]), ("", ""))[0] or None
    h = rec["h"]
    if math.isfinite(h):
        rec["ceiling_hard"] = float(NG.momentum_ceiling_a2(h, epsilon=float(conf["epsilon_hard"])))
        rec["ceiling_realistic"] = float(NG.momentum_ceiling_a2(
            h, epsilon=float(conf["epsilon_realistic"])))
        rec["a2_expected_yarkovsky"] = float(conf["epsilon_median_measured"]) * float(
            NG.momentum_ceiling_a2(h, albedo=NG.ALBEDO_TYPICAL,
                                   rho_kg_m3=NG.RHO_TYPICAL_KG_M3, epsilon=1.0))
        a2 = rec.get("a2", float("nan"))
        if math.isfinite(_f(a2)):
            rec["ratio_hard"] = abs(a2) / rec["ceiling_hard"]
            rec["ratio_realistic"] = abs(a2) / rec["ceiling_realistic"]
            rec["ratio_expected"] = abs(a2) / rec["a2_expected_yarkovsky"]
            rec["epsilon_eff"] = abs(a2) / float(NG.momentum_ceiling_a2(
                h, albedo=NG.ALBEDO_TYPICAL, rho_kg_m3=NG.RHO_TYPICAL_KG_M3, epsilon=1.0))
    else:
        for k in ("ceiling_hard", "ceiling_realistic", "a2_expected_yarkovsky",
                  "ratio_hard", "ratio_realistic", "ratio_expected", "epsilon_eff"):
            rec[k] = float("nan")
    # A SECOND ceiling, from the MEASURED diameter where SBDB has one.
    #
    # The gate above is deliberately generous: H with a high albedo and a low
    # density, so that exceeding it is a statement about the object and not
    # about the assumptions.  That generosity costs a factor of several, and an
    # object with a radiometric diameter does not need it --- H + albedo is a
    # stand-in for a size that has, for those objects, actually been measured.
    # So the measured-diameter ceiling is reported ALONGSIDE, never in place of,
    # the generous one: it is what vets an exceedance, not what declares it.
    # `rho` stays at the generous floor because a measured diameter says nothing
    # about the interior.
    d_km = _f(rec.get("diameter_km"))
    if math.isfinite(d_km) and d_km > 0:
        lvl = float(NG.momentum_ceiling_si(
            NG.amr_sphere(NG.RHO_GENEROUS_KG_M3, d_km * 1e3), r_au=1.0,
            epsilon=float(conf["epsilon_hard"])) * NG.SI_TO_AU_PER_DAY2)
        rec["ceiling_measured_diameter"] = lvl
        a2m = _f(rec.get("a2"))
        rec["ratio_measured_diameter"] = (abs(a2m) / lvl
                                          if math.isfinite(a2m) and lvl > 0 else float("nan"))
    else:
        rec["ceiling_measured_diameter"] = float("nan")
        rec["ratio_measured_diameter"] = float("nan")
    return rec


def screen_record(rec: dict, conf: dict, binaries: BinaryCatalogue | None = None) -> dict:
    """Assign the tier, with reasons and vetoes as separate fields."""
    reasons: list[str] = []
    vetoes: list[str] = []
    tier = "untestable"
    a2 = _f(rec.get("a2"))
    snr = _f(rec.get("a2_snr"))
    if rec.get("verdict") not in ("FITTED",) or not math.isfinite(a2) or not math.isfinite(snr):
        reasons.append(str(rec.get("reason") or rec.get("verdict") or "no_fit"))
        rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
        return rec
    if not math.isfinite(_f(rec.get("h"))):
        reasons.append("no_H__ceiling_undefined")
        rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
        return rec
    absorbed = _f(rec.get("a2_absorbed_fraction"))
    if math.isfinite(absorbed) and absorbed > float(conf["max_absorbed_fraction"]):
        reasons.append(f"absorbed_fraction_{absorbed:.4f}_above_{conf['max_absorbed_fraction']}")
        rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
        return rec
    if rec.get("nuisance_incomplete"):
        reasons.append("orbit_error_subspace_incomplete")
        rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
        return rec
    # Testable from here on.
    tier = "ordinary"
    ratio_hard = _f(rec.get("ratio_hard"))
    ratio_real = _f(rec.get("ratio_realistic"))
    detected = snr >= float(conf["min_snr_detection"])
    rec["a2_detected"] = bool(detected)
    if not detected:
        reasons.append(f"a2_snr_{snr:.1f}_below_{conf['min_snr_detection']}")
        rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
        return rec
    if ratio_real < 1.0:
        reasons.append("within_realistic_thermal_recoil_envelope")
        rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
        return rec
    tier = "watch"
    if ratio_hard < 1.0:
        reasons.append("above_realistic_envelope_below_hard_ceiling")
    else:
        reasons.append(f"above_hard_momentum_ceiling_x{ratio_hard:.1f}")
    # Vetoes: each is a named systematic that would explain the number.
    mv = rec.get("model_verdict")
    if mv == "GEOMETRIC_EXPLANATION_PREFERRED":
        vetoes.append(f"geometric_explanation_preferred:{rec.get('best_geometric_model')}")
    elif mv == "FAMILIES_DEGENERATE":
        vetoes.append("force_and_geometry_families_degenerate")
    elif mv != "FORCE_LAW_PREFERRED":
        vetoes.append(f"no_force_law_preferred:{mv}")
    es = _f(rec.get("excess_scatter"))
    if math.isfinite(es) and es > float(conf["max_excess_scatter"]):
        vetoes.append(f"excess_scatter_{es:.2f}_short_timescale_wobble_or_error_model")
    snr_p = (abs(a2) / _f(rec.get("a2_err_pessimistic"))
             if math.isfinite(_f(rec.get("a2_err_pessimistic"))) and _f(rec.get("a2_err_pessimistic")) > 0
             else snr)
    rec["a2_snr_pessimistic"] = snr_p
    if min(snr, snr_p) < float(conf["min_force_snr"]):
        vetoes.append(f"snr_{min(snr, snr_p):.1f}_below_{conf['min_force_snr']}_on_pessimistic_error_model")
    chi_al, chi_ac = _f(rec.get("chi_al")), _f(rec.get("chi_ac"))
    if math.isfinite(chi_al) and math.isfinite(chi_ac) and chi_ac > 3.0 * max(chi_al, 1.0):
        vetoes.append("across_scan_residual_also_anomalous__error_model_failure")
    if binaries is not None:
        from .screen import RejectionRecord

        probe = RejectionRecord(number_mp=float(rec["number_mp"]),
                                denomination=rec.get("denomination"))
        if binaries.match(probe):
            rec["known_binary"] = True
            vetoes.append("catalogued_binary_candidate_liberato2024")
    tdc = _f(rec.get("timing_delta_chi2"))
    fdc = _f(rec.get("a2_delta_chi2"))
    if math.isfinite(tdc) and math.isfinite(fdc) and tdc >= 0.8 * fdc:
        vetoes.append("timing_column_explains_as_much_as_the_force")
    if _f(rec.get("n_transits")) < 2 * int(conf["min_transits"]):
        reasons.append("few_transits_for_a_curvature")
    if ratio_hard >= 1.0 and not vetoes:
        tier = "interest"
        if rec.get("law_verdict") == "LAW_PREFERRED" and rec.get("best_law") == "force:constant":
            tier = "candidate"
            reasons.append("distance_independent_law_preferred")
    rec.update({"tier": tier, "reasons": reasons, "vetoes": vetoes})
    return rec


# ---------------------------------------------------------------------------
# Object selection and sharding (pure)
# ---------------------------------------------------------------------------
def choose_objects(numbers: list[int], sbdb: dict[int, dict], max_objects: int,
                   seed: int) -> list[int]:
    """Which objects to run when capped: every non-grav solution first, then a
    seeded random sample of the rest.  Uncapped: everything, sorted."""
    nums = sorted({int(n) for n in numbers})
    if not max_objects or max_objects <= 0 or max_objects >= len(nums):
        return nums
    controls = [n for n in nums if (sbdb.get(n) or {}).get("nongrav_fitted")]
    rest = [n for n in nums if n not in set(controls)]
    rng = np.random.default_rng(int(seed))
    k = max(0, int(max_objects) - len(controls))
    take = sorted(rng.choice(rest, size=min(k, len(rest)), replace=False).tolist()) if k else []
    return sorted(set(controls) | set(take))


def shard_slice(objects: list[int], shard: int, n_shards: int) -> list[int]:
    """Round-robin so every shard sees the same mix of bright and faint objects."""
    return [n for k, n in enumerate(objects) if k % max(int(n_shards), 1) == int(shard)]


def order_objects(objects: list[int], sbdb: dict[int, dict], seed: int) -> list[int]:
    """The order a shard actually works in: controls first, then a random shuffle.

    Two problems with running a shard in ascending ``number_mp``, and this fixes
    both.  (1) The positive controls --- the objects with a published, JPL-fitted
    ``A2`` --- are overwhelmingly NEAs, so they carry high numbers and would be
    reached last.  A shard that runs out of wall clock would then report an
    ``A2`` distribution with *no control on it*, and by this channel's own rule
    nothing in that output is believed.  Controls therefore go first, so the
    falsification test is paid for before anything else.  (2) Ascending number is
    ascending discovery date and therefore, to a good approximation, descending
    size: a truncated run in that order is a sample of large main-belt bodies,
    which is exactly the population in which the signal is weakest, and its
    element/pole statistics are not those of the catalogue.  A seeded shuffle of
    the non-controls makes any truncation an unbiased random subsample instead,
    so the population tests stay interpretable however early the run stops.

    Deterministic in ``seed`` and in the input list, so a re-run of the same
    shard does the same work in the same order and the per-chunk cache hits.
    """
    nums = [int(n) for n in objects]
    ctrl = [n for n in nums if (sbdb.get(n) or {}).get("nongrav_fitted")]
    seen = set(ctrl)
    rest = [n for n in nums if n not in seen]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(rest)).tolist() if rest else []
    return sorted(ctrl) + [rest[i] for i in perm]


def chunked(seq: list, size: int) -> list[list]:
    size = max(int(size), 1)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ---------------------------------------------------------------------------
# Runner-side catalogue helpers
# ---------------------------------------------------------------------------
def load_perturbers(paths: Paths, client: E.HorizonsClient | None, log=print,
                    jd_hi: float | None = None) -> E.PerturberSet:
    """The perturber grid, spanning the Gaia window AND out to ``jd_hi``.

    ``jd_hi`` is the latest SBDB osculation epoch (:func:`E.perturber_window`):
    the integrator starts there, so a grid that stops at the Gaia window's end
    is extrapolated for the whole stretch in between.  A cached grid that does
    not reach ``jd_hi`` is re-fetched, not reused.
    """
    lo = E.WINDOW_JD[0]
    hi = max(E.WINDOW_JD[1], float(jd_hi) if jd_hi is not None else E.WINDOW_JD[1])
    p = paths.work / "perturbers.npz"
    if p.exists():
        ps = E.PerturberSet.load(p)
        if ps.covers(lo, hi) and ps.n_bodies == len(E.PERTURBERS):
            return ps
    if client is None:
        raise E.EphemerisError("no perturber cache and no Horizons client")
    log(f"fetching perturber grids from Horizons over JD {lo:.1f}..{hi:.1f}")
    ps = E.PerturberSet.from_horizons(client, jd_lo=lo, jd_hi=hi,
                                      on_body=lambda b, n: log(f"  {b}: {n} nodes"))
    ps.save(p)
    return ps


def load_sbdb(paths: Paths, numbers_needed: list[int] | None = None, log=print) -> dict:
    """The SBDB catalogue, cached as parquet in the work dir."""
    import pandas as pd

    p = paths.work / "sbdb.parquet"
    meta = paths.work / "sbdb_meta.json"
    if p.exists():
        df = pd.read_parquet(p)
        rows = {int(r["number_mp"]): r for r in frame_to_rows(df)}
        for r in rows.values():
            r["nongrav_fitted"] = bool(r.get("nongrav_fitted"))
        return {"rows": rows, "meta": json.loads(meta.read_text()) if meta.exists() else {}}
    log("pulling the numbered-asteroid catalogue from JPL SBDB (full precision)")
    t0 = time.time()
    got = E.sbdb_bulk()
    info = {k: v for k, v in got.items() if k != "rows"}
    info["seconds"] = time.time() - t0
    rows = got["rows"]
    if not rows and numbers_needed:
        log(f"bulk SBDB failed ({got.get('error') or got.get('status')}); "
            f"falling back to per-object sbdb.api for {len(numbers_needed)} objects")
        info["fallback"] = "sbdb_single"
        for n in numbers_needed:
            try:
                rows[int(n)] = E.sbdb_single(int(n))
            except Exception as exc:                          # noqa: BLE001
                info.setdefault("failures", []).append(f"{n}: {exc}"[:120])
            time.sleep(0.25)
    if rows:
        df = pd.DataFrame(list(rows.values()))
        df.to_parquet(p, index=False)
        meta.write_text(json.dumps(info, indent=1, default=str))
    log(f"SBDB: {len(rows)} orbits in {info['seconds']:.0f} s "
        f"(dropped fields: {info.get('dropped_fields')})")
    return {"rows": rows, "meta": info}


def load_gaia_objects(paths: Paths, gaia, release: str, log=print) -> list[dict]:
    p = paths.work / f"objects_{release}.json"
    if p.exists():
        return json.loads(p.read_text())
    res = gaia.object_numbers(release=release)
    if res.verdict != "OK":
        raise RuntimeError(f"object list unavailable: {res.verdict} {res.notes}")
    rows = [{"number_mp": int(r["number_mp"]), "denomination": r.get("denomination")}
            for r in res.rows if r.get("number_mp") is not None]
    p.write_text(json.dumps(rows))
    log(f"Gaia {release}: {len(rows)} numbered objects")
    return rows


def fetch_chunk(gaia, numbers: list[int], release: str, paths: Paths, tag: str,
                log=print) -> tuple[dict[int, dict], dict]:
    """Every observation of the chunk's objects, cached as parquet."""
    import pandas as pd

    p = paths.work / "obs" / f"{tag}.parquet"
    info: dict = {"tag": tag, "n_requested": len(numbers)}
    if p.exists():
        df = pd.read_parquet(p)
        rows = frame_to_rows(df)
        info["cached"] = True
    else:
        t0 = time.time()
        res = gaia.observations_for_objects(numbers=numbers, release=release)
        info.update({"verdict": res.verdict, "truncated": res.truncated,
                     "seconds": time.time() - t0, "n_rows": res.n})
        if res.verdict == "TRUNCATED":
            # Split and re-pull rather than accept a partial arc.
            rows = []
            for half in chunked(numbers, max(len(numbers) // 2, 1)):
                r2 = gaia.observations_for_objects(numbers=half, release=release)
                info.setdefault("splits", []).append({"n": len(half), "verdict": r2.verdict,
                                                      "n_rows": r2.n})
                if r2.verdict in ("OK", "EMPTY"):
                    rows.extend(r2.rows)
        elif res.verdict in ("OK", "EMPTY"):
            rows = res.rows
        else:
            rows = []
            info["error"] = "; ".join(res.notes)[:300]
            # The QUERY failed: nothing is known about these objects, which is
            # not the same statement as "Gaia has no rows for them".
            info["failed"] = True
        if rows:
            p.parent.mkdir(parents=True, exist_ok=True)
            rows_to_frame(rows).to_parquet(p, index=False)
    groups = group_observations(rows)
    info["n_objects_returned"] = len(groups)
    info["n_rows_returned"] = len(rows)
    return groups, info


# ---------------------------------------------------------------------------
# Stage: probe
# ---------------------------------------------------------------------------
def sky_difference_mas(eph_a: E.AlignedEphemeris, eph_b: E.AlignedEphemeris,
                       observer: np.ndarray) -> np.ndarray:
    """Angular separation on the sky (mas) between two predictions per epoch."""
    sa = eph_a(eph_a.jd_ref)
    sb = eph_b(eph_b.jd_ref)
    ua = sa[:, :3] - observer
    ub = sb[:, :3] - observer
    ua = ua / np.linalg.norm(ua, axis=1)[:, None]
    ub = ub / np.linalg.norm(ub, axis=1)[:, None]
    cross = np.linalg.norm(np.cross(ua, ub), axis=1)
    return np.arcsin(np.clip(cross, 0.0, 1.0)) * R.MAS_PER_RAD


def stage_probe(conf: dict, paths: Paths, log=print, gaia=None, client=None) -> dict:
    from .acquire import GaiaSSO

    rec: dict = {"stage": "probe", "started_utc": _utc(), "verdict": "NOT_RUN"}
    out_path = paths.results / "probe_ephemeris.json"

    def checkpoint():
        E.save_json(out_path, rec)

    gaia = gaia or GaiaSSO()
    client = client or E.HorizonsClient(min_interval=float(conf["horizons_min_interval_s"]))
    try:
        # SBDB FIRST: the perturber grid has to reach its osculation epoch.
        cat = load_sbdb(paths, log=log)
        sbdb = cat["rows"]
        rec["sbdb"] = cat["meta"]
        rec["sbdb"]["n_rows"] = len(sbdb)
        win = E.perturber_window(r.get("epoch_jd") for r in sbdb.values())
        pert = load_perturbers(paths, client, log=log, jd_hi=win[1])
        rec["perturbers"] = {"n_bodies": pert.n_bodies, "labels": pert.labels,
                             "t_grid": [float(pert.t_grid[0]), float(pert.t_grid[-1])],
                             "required_window": list(win),
                             "retrieved_utc": pert.retrieved_utc}
        checkpoint()
        objs = load_gaia_objects(paths, gaia, conf["release"], log=log)
        numbers = [o["number_mp"] for o in objs]
        denom = {o["number_mp"]: o.get("denomination") for o in objs}
        rec["n_gaia_objects"] = len(numbers)
        checkpoint()
        # Warm the two VizieR catalogues HERE, once, rather than letting four
        # shards and then `assess` each discover a VizieR outage separately.
        # Greenberg+2020 is the control that does not come from the fit being
        # tested, so whether it was reachable is a fact about the run and
        # belongs in the run's record, not in a log line in one shard.
        yark = load_yarkovsky_catalogue(paths, log=log)
        rec["yarkovsky_controls"] = {
            "n_rows": len(yark), "vizier": C.GREENBERG2020_VIZIER,
            "verdict": "OK" if yark else "VIZIER_UNREACHABLE__SBDB_CONTROLS_ONLY"}
        binaries = load_binaries(paths, log=log)
        rec["binaries"] = {"retrieved_utc": binaries.retrieved_utc if binaries else None,
                           "verdict": "OK" if binaries else "UNAVAILABLE__NO_BINARY_VETO"}
        checkpoint()
        # Sample: the brightest (lowest-numbered) objects, a random draw, and every
        # non-grav solution present, capped.
        rng = np.random.default_rng(int(conf["seed"]))
        n_sample = int(conf["probe_sample"])
        low = numbers[:10]
        ng = [n for n in numbers if (sbdb.get(n) or {}).get("nongrav_fitted")]
        rec["n_nongrav_in_gaia"] = len(ng)
        rest = [n for n in numbers if n not in set(low) | set(ng)]
        rnd = sorted(rng.choice(rest, size=min(max(n_sample - 10, 0), len(rest)),
                                replace=False).tolist())
        sample = sorted(set(low) | set(rnd) | set(ng[:20]))
        rec["sample"] = sample
        conv = conventions_from(conf["conventions"])
        groups, info = fetch_chunk(gaia, sample, conf["release"], paths, "probe", log=log)
        rec["gaia_pull"] = info
        checkpoint()
        # (a) Horizons vs integrator, per object.
        bundles_i, skipped = integrator_bundles(groups, sbdb, pert, conv,
                                                on_group=lambda g: log(f"  group {g}"))
        rec["integrator_skipped"] = {str(k): v for k, v in skipped.items()}
        comp: dict[str, dict] = {}
        bundles_h: dict[int, EphemBundle] = {}
        for n, cols in groups.items():
            try:
                bh = horizons_bundle(n, cols, client, pert, conv)
            except E.EphemerisError as exc:
                comp[str(n)] = {"error": str(exc)[:200]}
                continue
            bundles_h[n] = bh
            if n in bundles_i:
                Rg = np.column_stack([R._col(cols, c, bh.jd_obs.size)
                                      for c in ("x_gaia", "y_gaia", "z_gaia")])
                d = sky_difference_mas(bundles_i[n].target_state, bh.target_state, Rg)
                d = d[np.isfinite(d)]
                comp[str(n)] = {"n": int(d.size),
                                "median_mas": float(np.median(d)) if d.size else None,
                                "max_mas": float(np.max(d)) if d.size else None,
                                "nongrav_fitted": bool((sbdb.get(n) or {}).get("nongrav_fitted"))}
            checkpoint()
        rec["integrator_vs_horizons"] = comp
        # Objects WITHOUT a non-grav term are the honest comparison: with one, the
        # integrator (gravity-only) and Horizons (with A2) are meant to differ.
        maxes = [c["max_mas"] for k, c in comp.items()
                 if c.get("max_mas") is not None and not c.get("nongrav_fitted")]
        meds = [c["median_mas"] for k, c in comp.items()
                if c.get("median_mas") is not None and not c.get("nongrav_fitted")]
        rec["integrator_agreement"] = {
            "n_objects": len(maxes),
            "median_of_max_mas": float(np.median(maxes)) if maxes else None,
            "p95_of_max_mas": float(np.percentile(maxes, 95)) if maxes else None,
            "median_of_median_mas": float(np.median(meds)) if meds else None,
            "threshold_mas": float(conf["max_integrator_disagreement_mas"]),
        }
        ng_diffs = [c["max_mas"] for k, c in comp.items()
                    if c.get("max_mas") is not None and c.get("nongrav_fitted")]
        rec["integrator_agreement"]["nongrav_objects_max_mas"] = ng_diffs
        agree = (maxes and rec["integrator_agreement"]["p95_of_max_mas"]
                 <= float(conf["max_integrator_disagreement_mas"]))
        rec["route_decision"] = ("integrator" if agree else "horizons"
                                 if maxes else "UNDECIDED_NO_COMPARISON")
        checkpoint()
        # (b) Conventions, measured on the lowest-numbered objects via Horizons.
        cands = [R.EpochConvention("epoch", "TCB", R.GAIA_JD_ZERO),
                 R.EpochConvention("epoch", "TDB", R.GAIA_JD_ZERO),
                 R.EpochConvention("epoch_utc", "UTC", R.GAIA_JD_ZERO)]
        votes: dict[str, list[float]] = {}
        per_obj = []
        src = orbit_source_for("horizons", None)
        for n in [x for x in sample if x in bundles_h][:int(conf["probe_conventions_objects"])]:
            res = R.resolve_conventions(groups[n], bundles_h[n].target_state, src,
                                        sun_state=pert.sun_state, epoch_candidates=cands,
                                        allow_partial=True)
            per_obj.append({"number_mp": n, "verdict": res.get("verdict"),
                            "margin": res.get("margin"),
                            "best": (res.get("conventions") or {}),
                            "scan": res.get("scan_convention"),
                            "top": res.get("trials", [])[:4]})
            for t in res.get("trials", []):
                c = t["convention"]
                key = json.dumps({k: c[k] for k in ("epoch", "apply_light_time",
                                                    "apply_stellar_aberration",
                                                    "apply_solar_deflection")}, sort_keys=True)
                votes.setdefault(key, []).append(float(t["median_abs_al_mas"]))
            checkpoint()
        pooled = sorted(((float(np.median(v)), k) for k, v in votes.items() if v),
                        key=lambda kv: kv[0])
        conv_rec: dict = {"per_object": per_obj,
                          "pooled": [{"median_abs_al_mas": s, "convention": json.loads(k)}
                                     for s, k in pooled[:8]]}
        if len(pooled) >= 2 and pooled[0][0] > 0:
            margin = pooled[1][0] / pooled[0][0]
            best = json.loads(pooled[0][1])
            scan_votes = [p["scan"].get("north_to_east") for p in per_obj
                          if p.get("scan") and p["scan"].get("verdict") == "OK"]
            n2e = (sum(1 for s in scan_votes if s) >= len(scan_votes) / 2) if scan_votes else True
            conv_rec["margin"] = margin
            conv_rec["conventions"] = {**best, "scan_pa_north_to_east": bool(n2e),
                                       "resolved_by": "MEASURED_ON_DATA",
                                       "resolution_margin": margin}
            conv_rec["verdict"] = "RESOLVED" if margin >= 3.0 else "AMBIGUOUS"
            if margin < 3.0:
                conv_rec["note"] = ("the second-best convention is within a factor of "
                                    "three; the config default is kept and the shards "
                                    "will say so")
        else:
            conv_rec["verdict"] = "NOT_ENOUGH_TRIALS"
        E.save_json(paths.results / "conventions.json", conv_rec)
        rec["conventions"] = {k: v for k, v in conv_rec.items() if k != "per_object"}
        conv_use = load_conventions(paths, conf)
        # (c) End-to-end A2 from both routes on the sample.
        fits_i, fits_h = {}, {}
        for n, cols in groups.items():
            row = sbdb.get(n)
            if n in bundles_i:
                r_i, _ = fit_object(n, cols, bundles_i[n], row, pert, conv_use, conf,
                                    denomination=denom.get(n))
                fits_i[str(n)] = {k: v for k, v in r_i.items() if not k.startswith("_")}
            if n in bundles_h:
                try:
                    r_h, _ = fit_object(n, cols, bundles_h[n], row, pert, conv_use, conf,
                                        denomination=denom.get(n))
                    fits_h[str(n)] = {k: v for k, v in r_h.items() if not k.startswith("_")}
                except R.CircularOrbitSourceError as exc:
                    fits_h[str(n)] = {"verdict": "REFUSED", "reason": str(exc)[:160]}
        pairs = [(fits_i[k]["a2"], fits_h[k]["a2"], fits_i[k].get("a2_err"))
                 for k in fits_i if k in fits_h and _fin(fits_i[k].get("a2"))
                 and _fin(fits_h[k].get("a2"))]
        rec["a2_route_comparison"] = {
            "n": len(pairs),
            "median_abs_difference_sigma": (float(np.median([abs(a - b) / e for a, b, e in pairs
                                                             if _fin(e) and e > 0]))
                                            if pairs else None),
            "integrator": fits_i, "horizons": fits_h,
        }
        rec["verdict"] = "OK"
    except Exception as exc:                                  # noqa: BLE001
        rec["verdict"] = "PROBE_FAILED"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
        import traceback

        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["finished_utc"] = _utc()
    rec["horizons_calls"] = getattr(client, "calls", None)
    checkpoint()
    log(f"probe: {rec['verdict']} route={rec.get('route_decision')} "
        f"agreement={rec.get('integrator_agreement')}")
    return rec


# ---------------------------------------------------------------------------
# Stage: one shard (acquire + fit + screen)
# ---------------------------------------------------------------------------
CSV_COLUMNS = (
    "number_mp", "denomination", "route", "verdict", "tier", "n_obs", "n_usable",
    "n_rejected", "n_transits", "arc_days", "mjd_min", "mjd_max",
    "median_sigma_al_mas", "rms_al_mas", "rms_ac_mas", "chi_al", "chi_ac",
    "median_r_au", "median_delta_au",
    "a2", "a2_err", "a2_err_pessimistic", "a2_snr", "a2_snr_pessimistic",
    "a2_absorbed_fraction", "a2_chi2_reduced", "a2_delta_chi2", "excess_scatter",
    "a1", "a1_err", "a1_snr", "a3", "a3_err", "a3_snr", "a2_joint", "a2_joint_err",
    "model_verdict", "best_model", "family_margin", "delta_chi2_force",
    "delta_chi2_geometric", "delta_chi2_sublimation", "delta_chi2_constant",
    "law_verdict", "best_law", "law_margin", "timing_dt_s", "timing_delta_chi2",
    "h", "albedo", "diameter_km", "a", "e", "i", "node", "argperi", "q",
    "orbit_class", "neo", "n_opp", "data_arc", "rms", "condition_code",
    "jpl_a2", "jpl_a2_sigma", "jpl_a2_snr", "jpl_nongrav_fitted", "is_control",
    "published_control", "ceiling_hard", "ceiling_realistic",
    "a2_expected_yarkovsky", "ratio_hard", "ratio_realistic", "ratio_expected",
    "ceiling_measured_diameter", "ratio_measured_diameter",
    "epsilon_eff", "a2_detected", "known_binary", "reasons", "vetoes", "reason",
)


def _csv_value(v) -> str:
    """One cell's TEXT.  Quoting is the csv module's job, not this function's.

    This used to quote a scalar string containing a comma and leave list cells
    bare, on the assumption that a list of reason tokens never holds a comma.
    It does: ``screen_record`` copies a failed object's ``reason`` --- usually
    ``f"{type(exc).__name__}: {exc}"`` --- into ``reasons``, and exception text
    is full of commas (array shapes, tuples).  Run 35746692260 wrote 4250 rows
    that way; the first failure on line 3 had 84 fields against an 83-column
    header, ``pd.read_csv`` refused the whole file, and ``assess`` reported
    four hours of fitting as NO_DATA_REACHED.  Every cell now goes through
    :func:`csv.writer`, which quotes commas, quotes and newlines wherever they
    occur.
    """
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    if isinstance(v, float):
        return "" if not math.isfinite(v) else repr(float(v))
    return str(v)


def write_shard_csv(path: Path, records: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in records:
        w.writerow([_csv_value(r.get(c)) for c in CSV_COLUMNS])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a job killed mid-write must leave the previous
    # checkpoint readable, not a truncated gzip stream.
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    tmp.replace(path)


_NUMPY_REPR = re.compile(r"^np\.(?:float|int|uint)\d*\((.*)\)$")


def repair_legacy_shard_text(text: str) -> tuple[str, dict]:
    """Re-serialise a shard CSV written by the pre-fix writer, losslessly.

    The old writer's only defect was an unquoted list cell, and the only list
    cell that can carry free text is ``reasons`` (``vetoes`` tokens are built
    from fixed prefixes plus a model name).  A row with too many fields
    therefore has every column before ``reasons`` intact, and ``vetoes`` and
    ``reason`` as its last two fields (``reason`` WAS quoted by the old writer);
    the overflow belongs to ``reasons`` and is joined back with the commas it
    lost.  A row broken across lines by an unquoted newline is short, and is
    re-joined with the following line(s) until it has its fields.

    The same writer passed a ``numpy.float64`` to ``repr``, which under numpy 2
    is ``np.float64(1.5e-14)`` rather than ``1.5e-14``: such a cell is unwrapped
    to the number it spells, since leaving it would turn the whole numeric
    column into text.

    A row that still cannot be put back to the header's width is DROPPED and
    counted --- never guessed at --- and the count is reported.
    """
    rows = list(csv.reader(io.StringIO(text)))
    stats = {"n_rows_in": max(len(rows) - 1, 0), "n_repaired_overflow": 0,
             "n_rejoined_newline": 0, "n_numpy_repr_cells": 0, "n_dropped": 0}
    if not rows:
        return text, stats
    header = rows[0]
    ncol = len(header)
    try:
        i_reasons = header.index("reasons")
    except ValueError:
        i_reasons = None
    out = io.StringIO()
    w = csv.writer(out, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(header)
    k = 1
    while k < len(rows):
        row = rows[k]
        k += 1
        if not row:
            continue
        while len(row) < ncol and k < len(rows) and i_reasons is not None:
            nxt = rows[k]
            k += 1
            row = row[:-1] + [row[-1] + "\n" + (nxt[0] if nxt else "")] + nxt[1:]
            stats["n_rejoined_newline"] += 1
        if len(row) > ncol and i_reasons is not None and i_reasons == ncol - 3:
            row = (row[:i_reasons] + [",".join(row[i_reasons:len(row) - 2])]
                   + row[len(row) - 2:])
            stats["n_repaired_overflow"] += 1
        if len(row) != ncol:
            stats["n_dropped"] += 1
            continue
        for j, cell in enumerate(row):
            m = _NUMPY_REPR.match(cell)
            if m:
                row[j] = m.group(1)
                stats["n_numpy_repr_cells"] += 1
        w.writerow(row)
    stats["n_rows_out"] = stats["n_rows_in"] - stats["n_dropped"]
    return out.getvalue(), stats


def read_shard_csv(f: Path):
    """One shard file as a DataFrame, repairing a pre-fix file if it must.

    Returns ``(frame, repair_stats_or_None)``.  A strict parse is tried first;
    only a file the strict parser refuses goes through the repair, so a
    well-formed file is read exactly as before.
    """
    import pandas as pd

    with gzip.open(f, "rt", newline="") if str(f).endswith(".gz") else open(
            f, newline="") as fh:
        text = fh.read()
    if "np.float" not in text and "np.int" not in text:
        try:
            return pd.read_csv(io.StringIO(text), low_memory=False), None
        except (pd.errors.ParserError, ValueError):
            pass
    fixed, stats = repair_legacy_shard_text(text)
    return pd.read_csv(io.StringIO(fixed), low_memory=False), stats


def read_shard_csvs(paths: Paths, log=print, repairs: dict | None = None):
    import pandas as pd

    files = sorted((paths.results / "fits").glob("shard_*.csv.gz"))
    frames = []
    for f in files:
        try:
            df, stats = read_shard_csv(f)
            if stats is not None:
                log(f"assess: {f.name} was malformed (pre-fix writer); repaired: {stats}")
                if repairs is not None:
                    repairs[f.name] = stats
            df["shard_file"] = f.name
            frames.append(df)
        except Exception as exc:                              # noqa: BLE001
            log(f"assess: could not read {f}: {exc}")
            if repairs is not None:
                repairs[f.name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    if not frames:
        return pd.DataFrame(), files
    df = pd.concat(frames, ignore_index=True)
    # A later re-run of the same shard supersedes an earlier one for its objects.
    df = df.drop_duplicates(subset=["number_mp"], keep="last")
    return df, files


def load_binaries(paths: Paths, log=print) -> BinaryCatalogue | None:
    """VizieR J/A+A/688/A50 (Liberato+2024) if reachable, cached as JSON."""
    p = paths.results / "binaries_liberato2024.json"
    if p.exists():
        rec = json.loads(p.read_text())
        return BinaryCatalogue(rows=rec.get("rows", []), retrieved_utc=rec.get("retrieved_utc"))
    try:
        from astroquery.vizier import Vizier

        v = Vizier(row_limit=-1)
        tabs = v.get_catalogs("J/A+A/688/A50")
        rows = []
        for t in tabs:
            cols = [c for c in t.colnames]
            for r in t:
                d = {}
                for c in cols:
                    val = r[c]
                    try:
                        d[c] = val.item() if hasattr(val, "item") else val
                    except Exception:                          # noqa: BLE001
                        d[c] = str(val)
                    if isinstance(d[c], bytes):
                        d[c] = d[c].decode("utf-8", "replace")
                    if isinstance(d[c], float) and not math.isfinite(d[c]):
                        d[c] = None
                # Normalise the number column name for BinaryCatalogue.match.
                for k in list(d):
                    if k.lower() in ("number", "num", "no", "nmp", "number_mp", "mp"):
                        d["number_mp"] = d[k]
                    if k.lower() in ("name", "designation", "desig", "object"):
                        d["denomination"] = d[k]
                rows.append(d)
        rec = {"rows": rows, "retrieved_utc": _utc(), "n_rows": len(rows),
               "reference": BinaryCatalogue().reference}
        E.save_json(p, rec)
        log(f"binary catalogue: {len(rows)} rows from VizieR")
        return BinaryCatalogue(rows=rows, retrieved_utc=rec["retrieved_utc"])
    except Exception as exc:                                  # noqa: BLE001
        log(f"binary catalogue unavailable: {type(exc).__name__}: {exc}"[:200])
        return None


def load_yarkovsky_catalogue(paths: Paths, log=print) -> dict[int, dict]:
    """Greenberg+2020 (AJ 159, 92), VizieR `J/AJ/159/92`, cached as JSON.

    A control value **independent of JPL's current solution**.  The SBDB ``A2``
    is the right primary control --- it is the same physical quantity on a
    longer arc with the covariance the fit actually has --- but it is a fit that
    saw Gaia DR2/DR3 astrometry.  Greenberg et al. published ``da/dt`` from
    247 objects' optical and radar astrometry, and :func:`controls.a2_from_dadt`
    maps it to ``A2`` exactly for JPL's ``g(r)``.  Where the two disagree, the
    disagreement is itself information about the control and is reported.

    Unreachable VizieR returns ``{}``: the run then scores its controls against
    SBDB alone and says so.  It never invents a published value.
    """
    p = paths.results / "controls_greenberg2020.json"
    if p.exists():
        try:
            rec = json.loads(p.read_text())
            # An EMPTY parse is not a catalogue: runs 35746692260 and later
            # cached 0 rows and re-used them forever.  Such a record is a miss.
            if rec.get("verdict") != "EMPTY_PARSE" and not (
                    rec.get("retrieved_utc") and not rec.get("rows")):
                return {int(k): v for k, v in (rec.get("rows") or {}).items()}
        except Exception:                                      # noqa: BLE001
            pass
    try:
        from astroquery.vizier import Vizier

        tabs = Vizier(row_limit=-1).get_catalogs(C.GREENBERG2020_VIZIER)
        rows: dict[int, dict] = {}
        seen_tables = [{"n_rows": len(t), "columns": list(t.colnames)} for t in tabs]
        for t in tabs:
            names = {c.lower(): c for c in t.colnames}
            num_c = next((names[k] for k in ("number", "num", "no", "mp", "aster",
                                              "ast", "n", "mpc", "nmp")
                          if k in names), None)
            if num_c is None:
                # A designation column like "(101955) Bennu" still carries the number.
                num_c = next((names[k] for k in ("name", "object", "desig", "asteroid",
                                                  "designation") if k in names), None)
            # da/dt is tabulated as `dadt` or `da/dt`; its sigma as `e_dadt`.
            dadt_c = next((c for c in t.colnames
                           if c.lower().replace("/", "").replace("_", "") == "dadt"), None)
            err_c = next((c for c in t.colnames
                          if c.lower().replace("/", "").replace("_", "") in
                          ("edadt", "sigmadadt", "sdadt")), None)
            if num_c is None or dadt_c is None:
                continue
            for r in t:
                try:
                    n = int(r[num_c])
                except (TypeError, ValueError):
                    m = re.match(r"\s*\(?(\d+)\)?", str(r[num_c]))
                    if not m:
                        continue
                    n = int(m.group(1))
                d = {"dadt_1e4_au_per_myr": _f(r[dadt_c]),
                     "dadt_sigma_1e4_au_per_myr": _f(r[err_c]) if err_c else float("nan")}
                for extra, key in (("a", "a_au"), ("e", "e"), ("H", "h")):
                    if extra in names.values() or extra in t.colnames:
                        d[key] = _f(r[extra])
                if math.isfinite(d["dadt_1e4_au_per_myr"]):
                    rows[n] = d
        rec = {"rows": {str(k): v for k, v in rows.items()}, "retrieved_utc": _utc(),
               "n_rows": len(rows), "vizier": C.GREENBERG2020_VIZIER,
               "reference": "Greenberg, Margot, Verma, Taylor & Hodge 2020, AJ 159, 92",
               "verdict": "OK" if rows else "EMPTY_PARSE",
               # What VizieR actually served, so an empty parse can be diagnosed
               # from the committed record instead of guessed at.
               "tables_seen": seen_tables}
        E.save_json(p, rec)
        log(f"Yarkovsky control catalogue: {len(rows)} da/dt values from VizieR "
            f"{C.GREENBERG2020_VIZIER}")
        return rows
    except Exception as exc:                                  # noqa: BLE001
        log(f"Yarkovsky control catalogue unavailable: {type(exc).__name__}: {exc}"[:200])
        return {}


def stage_shard(conf: dict, paths: Paths, shard: int, n_shards: int, *,
                log=print, gaia=None, client=None, route: str | None = None,
                max_objects: int | None = None, commit_hook=None,
                budget_minutes: float | None = None, now=time.time,
                sleep_fn=time.sleep) -> dict:
    from .acquire import GaiaSSO

    tag = f"shard_{int(shard)}_of_{int(n_shards)}"
    rec: dict = {"stage": "fit", "shard": int(shard), "n_shards": int(n_shards),
                 "started_utc": _utc(), "verdict": "NOT_RUN", "chunks": [],
                 "funnel": {}}
    # A clock INSIDE the job's own cap.  A job killed by `timeout-minutes` is
    # cancelled, and a cancelled job does not reliably run its `if: always()`
    # upload --- so an overrun would throw away every chunk the shard had
    # already fitted and checkpointed.  Stopping ourselves a little early turns
    # that into a clean, honest partial result that still reaches `assess`.
    t_start = now()
    budget_s = (float(budget_minutes) * 60.0
                if budget_minutes is not None and float(budget_minutes) > 0 else None)
    rec["budget_minutes"] = float(budget_minutes) if budget_s else None
    out_json = paths.results / "fits" / f"{tag}.json"
    out_csv = paths.results / "fits" / f"{tag}.csv.gz"
    gaia = gaia or GaiaSSO()
    client = client or E.HorizonsClient(min_interval=float(conf["horizons_min_interval_s"]))
    records: list[dict] = []
    details: dict[str, dict] = {}
    series_for_timing: list[R.ResidualSeries] = []

    def checkpoint(final: bool = False):
        rec["n_records"] = len(records)
        rec["funnel"] = funnel(records)
        rec["details_n"] = len(details)
        rec["horizons_calls"] = client.calls
        rec["gaia_calls"] = gaia.calls
        if final:
            rec["details"] = details
        write_shard_csv(out_csv, records)
        E.save_json(out_json, rec)
        if commit_hook is not None and not final:
            commit_hook()

    try:
        # Route: the probe's decision unless overridden.
        probe_path = paths.results / "probe_ephemeris.json"
        want = route or conf["ephemeris_route"]
        if want == "auto":
            if probe_path.exists():
                pr = json.loads(probe_path.read_text())
                want = pr.get("route_decision") or "integrator"
                rec["route_from_probe"] = want
            else:
                want = "integrator"
                rec["route_note"] = "no probe record; integrator route assumed"
        if want not in ("integrator", "horizons"):
            want = "integrator"
        rec["route"] = want
        conv = load_conventions(paths, conf)
        rec["conventions"] = conv.as_dict()
        cat = load_sbdb(paths, numbers_needed=None, log=log)
        sbdb = cat["rows"]
        win = E.perturber_window(r.get("epoch_jd") for r in sbdb.values())
        pert = load_perturbers(paths, client, log=log, jd_hi=win[1])
        if pert is not None:
            rec["perturber_grid"] = [float(pert.t_grid[0]), float(pert.t_grid[-1])]
        objs = load_gaia_objects(paths, gaia, conf["release"], log=log)
        numbers = [o["number_mp"] for o in objs]
        denom = {o["number_mp"]: o.get("denomination") for o in objs}
        cap = int(max_objects if max_objects is not None else conf["max_objects"])
        chosen = choose_objects(numbers, sbdb, cap, int(conf["seed"]))
        mine = order_objects(shard_slice(chosen, shard, n_shards), sbdb,
                             int(conf["seed"]) + int(shard))
        n_ctrl = sum(1 for n in mine if (sbdb.get(n) or {}).get("nongrav_fitted"))
        rec.update({"n_gaia_objects": len(numbers), "n_chosen": len(chosen),
                    "n_assigned": len(mine), "max_objects": cap,
                    "n_controls_assigned": n_ctrl,
                    "object_order": "controls_first_then_seeded_shuffle"})
        binaries = load_binaries(paths, log=log)
        rec["binary_catalogue"] = (binaries.retrieved_utc if binaries else None)
        checkpoint()
        chunks = chunked(mine, int(conf["objects_per_chunk"]))
        n_failed_run = 0
        for ci, chunk in enumerate(chunks):
            if budget_s is not None and now() - t_start > budget_s:
                rec["budget_stop"] = {"after_chunks": ci, "of_chunks": len(chunks),
                                      "elapsed_minutes": (now() - t_start) / 60.0,
                                      "note": ("stopped on the in-job clock so the "
                                               "checkpointed chunks survive; the "
                                               "remaining objects are UNMEASURED, "
                                               "not null")}
                log(f"{tag}: budget exhausted after {ci}/{len(chunks)} chunks")
                break
            t0 = time.time()
            groups, info = fetch_chunk(gaia, chunk, conf["release"], paths,
                                       f"{tag}_chunk{ci:04d}", log=log)
            if info.get("failed"):
                # One retry after a pause: TAP outages are often minutes long.
                sleep_fn(float(conf["failed_chunk_retry_sleep_s"]))
                groups, info = fetch_chunk(gaia, chunk, conf["release"], paths,
                                           f"{tag}_chunk{ci:04d}", log=log)
                info["retried"] = True
            info["chunk"] = ci
            failed = bool(info.get("failed"))
            n_failed_run = n_failed_run + 1 if failed else 0
            missing = [n for n in chunk if n not in groups]
            for n in missing:
                records.append({"number_mp": n, "denomination": denom.get(n),
                                "route": want,
                                "verdict": ("GAIA_QUERY_FAILED" if failed
                                            else "NO_OBSERVATIONS_RETURNED"),
                                "tier": "untestable",
                                "reason": (info.get("error") if failed else None),
                                "reasons": ([f"gaia_query_failed: {info.get('error')}"]
                                            if failed else ["no_rows_from_gaia"]),
                                "vetoes": []})
            if failed and n_failed_run >= int(conf["max_consecutive_failed_chunks"]):
                rec["chunks"].append(info)
                rec["archive_stop"] = {
                    "after_chunks": ci + 1, "of_chunks": len(chunks),
                    "consecutive_failed_chunks": n_failed_run,
                    "last_error": info.get("error"),
                    "note": ("the Gaia TAP service failed on every recent chunk; the "
                             "shard stopped rather than record an outage as absence")}
                log(f"{tag}: Gaia TAP failed on {n_failed_run} consecutive chunks; "
                    f"stopping ({info.get('error')})")
                checkpoint()
                break
            bundles: dict[int, EphemBundle] = {}
            skipped: dict[int, str] = {}
            if want == "integrator":
                bundles, skipped = integrator_bundles(
                    groups, sbdb, pert, conv,
                    on_group=lambda g: log(f"  propagated {g['n_objects']} objects "
                                           f"h={g['step_days']} in {g['seconds']:.0f} s"))
            else:
                for n, cols in groups.items():
                    try:
                        bundles[n] = horizons_bundle(n, cols, client, pert, conv)
                    except E.EphemerisError as exc:
                        skipped[n] = f"horizons: {exc}"[:160]
            # Controls get the pinned gravity-only route as well, on top of the
            # bulk route, because it is the cleanest measurement of their A2.
            n_fit = 0
            for n, cols in groups.items():
                row = sbdb.get(n)
                if n not in bundles:
                    r0 = {"number_mp": n, "denomination": denom.get(n), "route": want,
                          "verdict": "NO_EPHEMERIS", "reason": skipped.get(n, "unknown")}
                    annotate_orbit(r0, row, conf)
                    records.append(screen_record(r0, conf, binaries))
                    continue
                try:
                    r1, series = fit_object(n, cols, bundles[n], row, pert, conv, conf,
                                            denomination=denom.get(n))
                except R.CircularOrbitSourceError as exc:
                    r1 = {"number_mp": n, "denomination": denom.get(n), "route": want,
                          "verdict": "REFUSED_PROVENANCE", "reason": str(exc)[:160]}
                    series = None
                annotate_orbit(r1, row, conf)
                # A control ALWAYS gets the pinned gravity-only route, whatever
                # the bulk route.  On the Horizons route the bulk fit of every
                # control is refused as circular (Horizons integrates JPL's own
                # A2), so gating this on the integrator route --- as it was ---
                # meant a Horizons-route run could score no control at all: run
                # 35746692260 put all 78 controls in RESIDUALS_FAILED.  Where the
                # bulk fit was refused and the pinned fit succeeds, the pinned
                # fit IS the control's record, and the refusal is kept beside it.
                if (sbdb.get(n) or {}).get("nongrav_fitted"):
                    try:
                        pb = pinned_bundle(n, cols, client, pert, conv, sbdb)
                        r2, s2 = fit_object(n, cols, pb, row, pert, conv, conf,
                                            denomination=denom.get(n))
                        if r1.get("verdict") != "FITTED" and r2.get("verdict") == "FITTED":
                            bulk = {"bulk_route": r1.get("route"),
                                    "bulk_verdict": r1.get("verdict"),
                                    "bulk_reason": r1.get("reason")}
                            r1 = dict(r2)
                            r1.update(bulk)
                            annotate_orbit(r1, row, conf)
                            series = s2
                        r1["pinned_a2"] = _f(r2.get("a2"))
                        r1["pinned_a2_err"] = _f(r2.get("a2_err"))
                        r1["pinned_a2_snr"] = _f(r2.get("a2_snr"))
                        r1["pinned_absorbed"] = _f(r2.get("a2_absorbed_fraction"))
                        r1["_pinned_fits"] = r2.get("_fits")
                    except Exception as exc:                  # noqa: BLE001
                        r1["pinned_error"] = f"{type(exc).__name__}: {exc}"[:160]
                screen_record(r1, conf, binaries)
                fits = r1.pop("_fits", None)
                sep = r1.pop("_law_separability", None)
                pin = r1.pop("_pinned_fits", None)
                if r1.get("tier") in ("watch", "interest", "candidate") or r1.get("is_control"):
                    details[str(n)] = {"record": {k: v for k, v in r1.items()},
                                       "fits": fits, "law_separability": sep,
                                       "pinned_fits": pin}
                records.append(r1)
                if series is not None and r1.get("verdict") == "FITTED":
                    series_for_timing.append(series)
                    n_fit += 1
            info["n_fitted"] = n_fit
            info["seconds_total"] = time.time() - t0
            rec["chunks"].append(info)
            log(f"{tag} chunk {ci + 1}/{len(chunks)}: {len(groups)} objects, "
                f"{n_fit} fitted, {info['seconds_total']:.0f} s")
            checkpoint()
        # Whole-shard timing offset: the epoch-convention meter.
        rec["timing_offset"] = R.fit_common_time_offset(series_for_timing)
        if not records:
            rec["verdict"] = "NO_OBJECTS"
        elif rec.get("archive_stop"):
            rec["verdict"] = "ARCHIVE_UNREACHABLE_PARTIAL"
        elif rec.get("budget_stop"):
            rec["verdict"] = "OK_PARTIAL_BUDGET"
        else:
            rec["verdict"] = "OK"
    except Exception as exc:                                  # noqa: BLE001
        rec["verdict"] = "SHARD_FAILED"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
        import traceback

        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["finished_utc"] = _utc()
    checkpoint(final=True)
    log(f"{tag}: {rec['verdict']} funnel={rec.get('funnel')}")
    return rec


def funnel(records: list[dict]) -> dict:
    out: dict = {"n_records": len(records)}
    verdicts: dict[str, int] = {}
    tiers: dict[str, int] = {}
    for r in records:
        verdicts[str(r.get("verdict"))] = verdicts.get(str(r.get("verdict")), 0) + 1
        tiers[str(r.get("tier"))] = tiers.get(str(r.get("tier")), 0) + 1
    out["verdicts"] = verdicts
    out["tiers"] = tiers
    out["n_fitted"] = verdicts.get("FITTED", 0)
    out["n_detected_snr3"] = sum(1 for r in records if r.get("a2_detected"))
    out["n_controls"] = sum(1 for r in records if r.get("is_control"))
    return out


# ---------------------------------------------------------------------------
# Stage: assess (offline)
# ---------------------------------------------------------------------------
def strata_labels(df, n_bins: int = 4) -> np.ndarray:
    """Matching strata for the population null: H, transit count, arc length."""
    def qbin(x):
        x = np.asarray(x, dtype=float)
        good = np.isfinite(x)
        out = np.zeros(x.size, dtype=int)
        if good.sum() < n_bins:
            return out
        edges = np.unique(np.quantile(x[good], np.linspace(0, 1, n_bins + 1)[1:-1]))
        out[good] = np.searchsorted(edges, x[good])
        return out

    hb = qbin(df["h"].to_numpy(dtype=float))
    tb = qbin(df["n_transits"].to_numpy(dtype=float))
    ab = qbin(df["arc_days"].to_numpy(dtype=float))
    return hb * 100 + tb * 10 + ab


def _column(frame, name: str) -> np.ndarray:
    """A float column, or all-NaN if the shard's CSV predates it.

    `assess` gathers shards that may have been written by an earlier build of
    this module --- the workflow re-runs failed shards, and a run can be
    assessed from another run's artifacts.  A missing column must degrade that
    one statistic, not raise and lose the whole assessment.
    """
    if name in getattr(frame, "columns", ()):
        return frame[name].to_numpy(dtype=float)
    return np.full(len(frame), float("nan"))


def assess_frame(df, conf: dict, details: dict | None = None,
                 yarkovsky: dict[int, dict] | None = None) -> dict:
    """The assessment on the gathered per-object table (pure)."""
    out: dict = {"n_objects": int(len(df))}
    if len(df) == 0:
        out["verdict"] = "NO_DATA_REACHED"
        return out
    fitted = df[df["verdict"] == "FITTED"].copy()
    out["funnel"] = {
        "n_records": int(len(df)),
        "verdicts": {str(k): int(v) for k, v in df["verdict"].value_counts().items()},
        "tiers": {str(k): int(v) for k, v in df["tier"].value_counts().items()},
        "n_fitted": int(len(fitted)),
        "routes": {str(k): int(v) for k, v in df["route"].value_counts().items()},
    }
    snr = _column(fitted, "a2_snr")
    a2 = _column(fitted, "a2")
    err = _column(fitted, "a2_err")
    good = np.isfinite(snr) & np.isfinite(a2) & np.isfinite(err) & (err > 0)
    z = np.where(good, a2 / np.where(err > 0, err, np.nan), np.nan)
    z = z[np.isfinite(z)]
    dist: dict = {"n_with_a2": int(good.sum())}
    if z.size:
        mad = float(np.median(np.abs(z - np.median(z))) * 1.4826)
        dist.update({
            "z_median": float(np.median(z)), "z_mad_sigma": mad,
            "z_p16": float(np.percentile(z, 16)), "z_p84": float(np.percentile(z, 84)),
            "fraction_abs_z_above_3": float(np.mean(np.abs(z) > 3)),
            "expected_fraction_if_gaussian": 0.0027,
            "n_abs_z_above_3": int(np.sum(np.abs(z) > 3)),
            "n_abs_z_above_5": int(np.sum(np.abs(z) > 5)),
            "sign_negative_fraction_detected": (float(np.mean(a2[good & (np.abs(a2 / err) > 3)] < 0))
                                                if np.any(good & (np.abs(a2 / err) > 3)) else None),
            "median_a2_err_au_day2": float(np.median(err[good])),
            "p10_a2_err_au_day2": float(np.percentile(err[good], 10)),
        })
        exp = _column(fitted, "a2_expected_yarkovsky")
        ok = good & np.isfinite(exp) & (exp > 0)
        if ok.any():
            dist["median_sensitivity_over_yarkovsky_expectation"] = float(
                np.median(err[ok] / exp[ok]))
            dist["n_objects_with_sensitivity_below_expectation"] = int(
                np.sum(err[ok] < exp[ok]))
        absorbed = _column(fitted, "a2_absorbed_fraction")
        if np.any(np.isfinite(absorbed)):
            dist["absorbed_fraction_median"] = float(np.nanmedian(absorbed))
            dist["absorbed_fraction_p90"] = float(np.nanpercentile(absorbed, 90))
        ratio = _column(fitted, "ratio_hard")
        det = good & (np.abs(a2 / err) >= float(conf["min_snr_detection"]))
        dist["n_detected"] = int(det.sum())
        dist["n_detected_above_hard_ceiling"] = int(np.sum(det & (ratio >= 1.0)))
        dist["n_detected_above_realistic"] = int(np.sum(
            det & (_column(fitted, "ratio_realistic") >= 1.0)))
        eps = _column(fitted, "epsilon_eff")
        if np.any(det & np.isfinite(eps)):
            dist["epsilon_eff_detected_median"] = float(np.median(eps[det & np.isfinite(eps)]))
        # THE A2 DISTRIBUTION AGAINST THE YARKOVSKY EXPECTATION.
        #
        # `ratio_expected` is |A2| divided by what thermal recoil at LOOM's
        # measured median efficiency (epsilon = 0.074, 589 SBDB asteroids)
        # would give a body of that H.  A population of ordinary asteroids
        # detected at S/N >= 3 should sit AROUND 1 with a tail to a few: that
        # is Yarkovsky, and it is the null this channel is looking past.  A
        # percentile far above 1 is either an anomalous object or --- much more
        # likely, and to be excluded first --- an error model that is too
        # small.  Quoted as percentiles rather than a mean because the
        # distribution is heavy-tailed by construction.
        rexp = _column(fitted, "ratio_expected")
        dok = det & np.isfinite(rexp) & (rexp > 0)
        if dok.sum() >= 5:
            qs = np.percentile(rexp[dok], [10, 25, 50, 75, 90, 99])
            dist["ratio_to_yarkovsky_expectation_detected"] = {
                "n": int(dok.sum()),
                "p10": float(qs[0]), "p25": float(qs[1]), "median": float(qs[2]),
                "p75": float(qs[3]), "p90": float(qs[4]), "p99": float(qs[5]),
                "fraction_above_10x": float(np.mean(rexp[dok] > 10.0)),
                "epsilon_median_measured": float(conf["epsilon_median_measured"]),
                "note": ("|A2| over the thermal-recoil expectation at LOOM's measured "
                         "median efficiency; an ordinary Yarkovsky population sits "
                         "around 1, and a median far above it indicts the error model "
                         "before it indicts the sky"),
            }
        # Quality independence: does |z| track how well the object was observed?
        for col in ("n_transits", "arc_days", "median_sigma_al_mas", "h"):
            x = _column(fitted, col)
            m = good & np.isfinite(x)
            if m.sum() > 30:
                from scipy.stats import spearmanr

                rho = spearmanr(np.abs(a2[m] / err[m]), x[m]).correlation
                dist[f"spearman_abs_z_vs_{col}"] = float(rho)
    out["a2_distribution"] = dist
    # Controls.
    yark = yarkovsky or {}
    # An object is a control if JPL fits an A2 at S/N >= 3 OR Greenberg+2020
    # published a da/dt for it.  The second clause matters: it adds controls
    # whose published detection does not depend on the same fit being consulted
    # as the truth, and it keeps the estimator exercised if SBDB is unreachable.
    is_ctrl = fitted["is_control"].astype(float) > 0 if "is_control" in fitted else None
    if yark and "number_mp" in fitted:
        in_yark = fitted["number_mp"].astype(int).isin(list(yark)).to_numpy()
        is_ctrl = in_yark if is_ctrl is None else (is_ctrl.to_numpy() | in_yark)
    ctrl = fitted[is_ctrl] if is_ctrl is not None else fitted.iloc[0:0]
    scored = []
    for _, r in ctrl.iterrows():
        s = score_control(r.get("a2"), r.get("a2_err"), r.get("jpl_a2"), r.get("jpl_a2_sigma"),
                          min_snr=float(conf["min_snr_detection"]))
        s.update({"number_mp": int(r["number_mp"]), "denomination": r.get("denomination"),
                  "published": r.get("published_control") if isinstance(r.get("published_control"), str) else None,
                  "n_transits": _f(r.get("n_transits")), "arc_days": _f(r.get("arc_days")),
                  "absorbed_fraction": _f(r.get("a2_absorbed_fraction")),
                  "route": r.get("route")})
        if details and str(int(r["number_mp"])) in details:
            d = details[str(int(r["number_mp"]))].get("record", {})
            if _fin(d.get("pinned_a2")):
                s["pinned"] = score_control(d.get("pinned_a2"), d.get("pinned_a2_err"),
                                            r.get("jpl_a2"), r.get("jpl_a2_sigma"),
                                            min_snr=float(conf["min_snr_detection"]))
        # The independent published control, where Greenberg+2020 has one.
        g = yark.get(int(r["number_mp"]))
        if g:
            a_au = _f(g.get("a_au")) if _fin(g.get("a_au")) else _f(r.get("a"))
            ecc = _f(g.get("e")) if _fin(g.get("e")) else _f(r.get("e"))
            pa2 = C.a2_from_dadt(g.get("dadt_1e4_au_per_myr"), a_au, ecc)
            psig = abs(C.a2_from_dadt(g.get("dadt_sigma_1e4_au_per_myr"), a_au, ecc))
            s["published_a2"] = pa2
            s["published_a2_sigma"] = psig if math.isfinite(psig) and psig > 0 else float("nan")
            s["published_dadt_1e4_au_per_myr"] = _f(g.get("dadt_1e4_au_per_myr"))
            s["published_source"] = "Greenberg+2020 AJ 159, 92"
            if math.isfinite(pa2):
                s["vs_published"] = score_control(
                    r.get("a2"), r.get("a2_err"), pa2, s["published_a2_sigma"],
                    min_snr=float(conf["min_snr_detection"]))
                jp = _f(r.get("jpl_a2"))
                # Do the two *published* control values agree with each other?
                # If they do not, the control itself is in question before the fit is.
                s["published_vs_jpl_ratio"] = (pa2 / jp if math.isfinite(jp) and jp != 0
                                               else float("nan"))
        scored.append(s)
    controls = summarise_controls(scored)
    controls["scored"] = scored
    pinned = [s["pinned"] for s in scored if s.get("pinned")]
    if pinned:
        controls["pinned_route"] = summarise_controls(pinned)
    vs_pub = [s["vs_published"] for s in scored if s.get("vs_published")]
    if vs_pub:
        controls["against_greenberg2020"] = summarise_controls(vs_pub)
        ratios = [s["published_vs_jpl_ratio"] for s in scored
                  if _fin(s.get("published_vs_jpl_ratio"))]
        if ratios:
            controls["published_vs_jpl_ratio_median"] = float(np.median(ratios))
            controls["n_published_vs_jpl_compared"] = len(ratios)
    controls["n_greenberg2020_rows"] = len(yark)
    out["controls"] = controls
    # Exceedances and their vetting.
    exc_mask = (fitted["tier"].isin(["watch", "interest", "candidate"])
                & (fitted["ratio_hard"].astype(float) >= 1.0))
    exc = fitted[exc_mask].sort_values("ratio_hard", ascending=False)
    cols_keep = ["number_mp", "denomination", "tier", "a2", "a2_err", "a2_snr",
                 "a2_snr_pessimistic", "ratio_hard", "ratio_realistic",
                 "ratio_measured_diameter", "diameter_km", "albedo",
                 "h", "a", "e", "i", "n_transits",
                 "arc_days", "a2_absorbed_fraction", "excess_scatter", "model_verdict",
                 "best_model", "law_verdict", "best_law", "chi_al", "chi_ac",
                 "jpl_nongrav_fitted", "jpl_a2", "is_control", "known_binary",
                 "vetoes", "reasons", "route"]
    out["exceedances"] = [
        {k: (None if (isinstance(v, float) and not math.isfinite(v)) else
             (v.item() if hasattr(v, "item") else v)) for k, v in r.items() if k in cols_keep}
        for _, r in exc.iterrows()]
    out["n_exceedances"] = int(len(exc))
    out["n_interest"] = int((fitted["tier"] == "interest").sum())
    out["n_candidate"] = int((fitted["tier"] == "candidate").sum())
    # Population decision on the interest+candidate set against matched draws.
    pop_mask = fitted["tier"].isin(["interest", "candidate"]).to_numpy()
    rows = [{"a": _f(r.get("a")), "e": _f(r.get("e")), "i": _f(r.get("i")),
             "node": _f(r.get("node")), "h": _f(r.get("h")),
             "mjd_min": _f(r.get("mjd_min")), "mjd_max": _f(r.get("mjd_max"))}
            for _, r in fitted.iterrows()]
    labels = strata_labels(fitted) if len(fitted) else np.zeros(0, dtype=int)
    if len(fitted) >= int(conf["min_parent_for_population"]) and pop_mask.sum() >= int(
            conf["min_anomalies_for_population"]):
        out["population"] = replication_tests(rows, pop_mask, labels,
                                              n_null=int(conf["population_n_null"]),
                                              seed=int(conf["seed"]))
    else:
        out["population"] = {"verdict": "INSUFFICIENT_POPULATION",
                             "n_parent": int(len(fitted)), "n_anomaly": int(pop_mask.sum()),
                             "note": (f"needs >= {conf['min_parent_for_population']} fitted "
                                      f"objects and >= {conf['min_anomalies_for_population']} "
                                      f"objects above the hard ceiling with no veto")}
    # The verdict.
    n_gaia_failed = int((df["verdict"] == "GAIA_QUERY_FAILED").sum())
    out["n_gaia_query_failed"] = n_gaia_failed
    if len(fitted) == 0 and n_gaia_failed > 0:
        # Every record the archive answered produced no fit, and some it never
        # answered at all: that is an outage, and it keeps the NO_DATA_REACHED
        # stem so it cannot be committed over a measured summary.
        out["verdict"] = "NO_DATA_REACHED__GAIA_TAP_FAILED"
    elif len(fitted) == 0:
        out["verdict"] = "NO_OBJECT_FITTED"
    elif out["population"].get("verdict") == "REPLICATION_STRUCTURE_DETECTED":
        out["verdict"] = "REPLICATION_STRUCTURE_DETECTED__VET_BEFORE_BELIEVING"
    elif out["n_candidate"] > 0:
        out["verdict"] = f"CANDIDATES_{out['n_candidate']}__DISTANCE_INDEPENDENT_ABOVE_CEILING"
    elif out["n_interest"] > 0:
        out["verdict"] = f"CEILING_EXCEEDANCES_{out['n_interest']}__UNDER_VETTING"
    elif out["n_exceedances"] > 0:
        out["verdict"] = f"ALL_{out['n_exceedances']}_EXCEEDANCES_VETOED"
    else:
        out["verdict"] = "NO_CEILING_EXCEEDANCE"
    # A control verdict that says the estimator is WRONG --- not merely
    # insensitive --- is stamped onto the run's verdict, because a ceiling
    # exceedance found by an estimator that cannot reproduce a known Yarkovsky
    # A2 is a property of the estimator.  CONTROLS_BELOW_SENSITIVITY is not in
    # this list: it says the controls were too faint to exercise, which is a
    # statement about the sample and is carried in `controls` instead.
    if controls.get("verdict") in ("CONTROLS_FAILED_SIGN", "CONTROLS_INCONSISTENT"):
        out["verdict"] = "ESTIMATOR_FAILS_CONTROLS__" + out["verdict"]
    return out


def stage_assess(conf: dict, paths: Paths, log=print) -> dict:
    repairs: dict = {}
    df, files = read_shard_csvs(paths, log=log, repairs=repairs)
    details: dict = {}
    shard_meta = []
    for f in sorted((paths.results / "fits").glob("shard_*.json")):
        try:
            j = json.loads(f.read_text())
            details.update(j.get("details") or {})
            shard_meta.append({"file": f.name, "verdict": j.get("verdict"),
                               "route": j.get("route"), "n_records": j.get("n_records"),
                               "funnel": j.get("funnel"), "timing_offset": j.get("timing_offset"),
                               "conventions": j.get("conventions"),
                               "budget_stop": j.get("budget_stop"),
                               "n_assigned": j.get("n_assigned"),
                               "n_controls_assigned": j.get("n_controls_assigned"),
                               "started_utc": j.get("started_utc"),
                               "finished_utc": j.get("finished_utc")})
        except Exception as exc:                              # noqa: BLE001
            shard_meta.append({"file": f.name, "error": str(exc)[:120]})
    out = assess_frame(df, conf, details, yarkovsky=load_yarkovsky_catalogue(paths, log=log))
    n_rec_json = sum(int(m.get("n_records") or 0) for m in shard_meta)
    if out.get("n_objects", 0) == 0 and n_rec_json > 0:
        # The shards SAY they fitted objects and the table came back empty:
        # that is this pipeline failing, not the sky returning nothing, and the
        # verdict must not read as an empty measurement.  (It keeps the
        # NO_DATA_REACHED stem so commit_results.sh still refuses to lay it
        # over a summary that carries a real measurement.)
        out["verdict"] = "NO_DATA_REACHED__SHARD_OUTPUT_UNREADABLE"
        out["note"] = (f"shard JSON reports {n_rec_json} records but no shard CSV row "
                       "could be read; see shard_csv_repairs")
    out["shards"] = shard_meta
    out["shard_files"] = [f.name for f in files]
    out["shard_csv_repairs"] = repairs
    out["assessed_utc"] = _utc()
    timing = [m["timing_offset"] for m in shard_meta if m.get("timing_offset")]
    dts = [t.get("dt_seconds") for t in timing if _fin(t.get("dt_seconds"))]
    out["timing_offset_seconds_by_shard"] = dts
    E.save_json(paths.results / "assessment.json", out)
    E.save_json(paths.results / "controls.json", out.get("controls", {}))
    E.save_json(paths.results / "exceedances.json",
                {"n": out.get("n_exceedances", 0), "rows": out.get("exceedances", [])})
    summary = {
        "channel": "sextant", "verdict": out["verdict"], "assessed_utc": out["assessed_utc"],
        "funnel": out.get("funnel"), "controls": {k: v for k, v in out.get("controls", {}).items()
                                                  if k != "scored"},
        "a2_distribution": out.get("a2_distribution"),
        "n_exceedances": out.get("n_exceedances"), "n_interest": out.get("n_interest"),
        "n_candidate": out.get("n_candidate"),
        "population_verdict": (out.get("population") or {}).get("verdict"),
        "routes": (out.get("funnel") or {}).get("routes"),
        "timing_offset_seconds_by_shard": dts,
        "coverage": {"n_shard_files": len(files),
                     "n_objects": out.get("n_objects", 0),
                     # A shard file the strict parser refused and the legacy
                     # repair had to rebuild (or could not read at all).  A
                     # file with records that yields n_objects == 0 is a
                     # pipeline defect, and this field is where it shows.
                     "shard_csv_repairs": repairs,
                     "n_records_in_shard_json": sum(int(m.get("n_records") or 0)
                                                    for m in shard_meta),
                     # How much of the assigned sample was actually reached.  A
                     # shard that stopped on its clock leaves objects UNMEASURED,
                     # and an unmeasured object is not a null --- the funnel
                     # counts below are a statement about what was fitted, not
                     # about the catalogue.
                     "n_assigned": sum(int(m.get("n_assigned") or 0) for m in shard_meta),
                     "n_shards_budget_stopped": sum(1 for m in shard_meta
                                                    if m.get("budget_stop")),
                     "shard_verdicts": sorted({str(m.get("verdict")) for m in shard_meta})},
    }
    E.save_json(paths.results / "summary.json", summary)
    log(f"assess: {out['verdict']} funnel={out.get('funnel')} controls={summary['controls']}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(stage: str, shard: str = "0/1", cfg=None, out_dir=None, work_dir=None,
        route: str | None = None, max_objects: int | None = None,
        n_shards_for_all: int = 1, release: str | None = None,
        budget_minutes: float | None = None, log=print) -> dict:
    conf = load_config(cfg)
    if max_objects is not None:
        conf["max_objects"] = int(max_objects)
    if release:
        if release not in ("gaiafpr", "gaiadr3"):
            raise ValueError(f"unknown release {release!r}; expected gaiafpr or gaiadr3")
        conf["release"] = release
    paths = Paths.make(out_dir, work_dir)
    try:
        i, n = (int(x) for x in str(shard).split("/"))
    except ValueError as exc:
        raise ValueError("--shard must be i/n, e.g. 3/16") from exc
    if stage == "probe":
        return stage_probe(conf, paths, log=log)
    if stage in ("acquire", "fit", "screen"):
        return stage_shard(conf, paths, i, n, log=log, route=route,
                           max_objects=max_objects, budget_minutes=budget_minutes)
    if stage == "assess":
        return stage_assess(conf, paths, log=log)
    if stage == "all":
        stage_probe(conf, paths, log=log)
        nsh = max(int(n_shards_for_all), 1)
        # In-process `all` splits one budget across the shards it runs.
        per = (float(budget_minutes) / nsh
               if budget_minutes is not None and float(budget_minutes) > 0 else None)
        for k in range(nsh):
            stage_shard(conf, paths, k, nsh, log=log, route=route,
                        max_objects=max_objects, budget_minutes=per)
        return stage_assess(conf, paths, log=log)
    raise ValueError(f"unknown stage {stage!r}")

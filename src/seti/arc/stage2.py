"""ARC stage 2 --- IS THE FLARE ON THE TARGET?  (docs/arc.md §6)

Stage 1 ends with stars whose catalogued flare energy exceeds the magnetic
energy their own spots can store, ``centroid: not_checked``.  The catalogues
cannot say which pixel flared; the Kepler target pixel file can.  For every
shortlisted star (interest tier first, then watch) this stage

1. re-fetches the star's flare rows from the catalogue (Yang & Liu 2019 gives
   the quarter and a start / end time; Tu+2022 a sector) and its stellar
   parameters from Berger+2020 (``J/AJ/159/280``) and Gaia DR3 FLAME
   (``I/355/paramp``), closing the ``stellar_params_assumed`` flag;
2. pulls the light curve and the target pixel file of every quarter that
   holds a flare above the ceiling, re-detects the flare, matches it to the
   catalogue row (by time window where the catalogue has one, by energy rank
   where it does not), re-measures its energy with one method and the
   quarter's own rotational amplitude;
3. runs the Gaia DR3 census inside one Kepler pixel (4") and the 12"
   aperture --- what fraction of the aperture flux each neighbour supplies and
   how much it would have to brighten to make the flare;
4. measures, per flare, the flux-weighted centroid shift in flare against the
   out-of-flare baseline and the centroid of the per-pixel-detrended
   difference image, and attributes the flare to the target or a neighbour
   (:mod:`seti.arc.pixels`);
5. recomputes xi on the measured radius / Teff with the star's own amplitude.

Verdict per star: ``flare_on_target`` / ``flare_on_neighbour`` /
``centroid_ambiguous`` / ``centroid_untestable``, with the xi recomputed on
measured parameters beside it.  A star whose pixel file or light curve did
not arrive is ``centroid_untestable`` with the reason; nothing about it is
inferred.

Everything network-facing is injectable (``query_fn``, ``cone_fn``, ``lc_fn``,
``tpf_fn``); the offline suite drives the whole stage through synthetic data.
Runs as ``python -m seti.arc.stage2 --stage all``.
"""

from __future__ import annotations

import argparse
import json
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..metronome.acquire import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    AcquisitionLog,
    _vizier_cone,
    resolve_columns,
    table_columns,
    tap_query,
)
from ..metronome.run import normalise_time_system
from ..metronome.windows import guess_time_system
from .acquire import ROLE_PATTERNS, _clean_id, fetch_star_params_by_id, resolve_arc_columns
from .ceiling import B_DEFAULT_G, DEFAULT_GEOMETRIC_FACTOR, energy_to_bolometric, star_ceiling
from .flares import (
    MATCH_NONE,
    detect_flares,
    flare_energy_band,
    flare_energy_shibayama,
    match_flares,
    rotational_amplitude,
)
from .pixels import (
    KEPLER_PIXEL_ARCSEC,
    KEPLER_PRF_SIGMA_PX,
    OUTCOME_UNTESTABLE,
    TESS_PIXEL_ARCSEC,
    TESS_PRF_SIGMA_PX,
    VERDICT_ON_NEIGHBOUR,
    VERDICT_ON_TARGET,
    VERDICT_UNTESTABLE,
    anchor_sources,
    attribute_flare,
    census,
    centroid_shift,
    difference_image,
    flare_cadence_masks,
    star_verdict,
)
from .run import load_arc_config

BKJD_OFFSET = 2454833.0
BTJD_OFFSET = 2457000.0
SPH_TO_RANGE = 2.0 * math.sqrt(2.0)   # verify: sinusoid peak-to-peak / standard deviation

VERDICT_S2_NO_DATA = "STAGE2_NO_DATA_REACHED"
VERDICT_S2_ON_TARGET = "CEILING_EXCESS_ON_TARGET_PENDING_SPECTROSCOPY"
VERDICT_S2_NEIGHBOUR = "CEILING_EXCESS_TRACED_TO_NEIGHBOUR"
VERDICT_S2_DISSOLVED = "CEILING_EXCESS_DISSOLVED_ON_MEASURED_PARAMETERS"
VERDICT_S2_UNTESTABLE = "CEILING_EXCESS_CENTROID_UNTESTABLE"
VERDICT_S2_AMBIGUOUS = "CEILING_EXCESS_CENTROID_AMBIGUOUS"

ROUTE_LIGHTKURVE = "lightkurve"
ROUTE_MAST_FITS = "astroquery_mast_fits"

FLAME_PATTERNS = {
    "gaia_id": [r"^source$", r"^source_?id$"],
    "radius_flame": [r"^rad(ius)?[-_]?flame$"],
    "mass_flame": [r"^mass[-_]?flame$"],
    "lum_flame": [r"^lum[-_]?flame$"],
    "age_flame": [r"^age[-_]?flame$"],
    "evol_flame": [r"^evol(stage)?[-_]?flame$", r"^evol$"],
    "teff_gspphot": [r"^teff$", r"^teff[-_]?gspphot$"],
    "logg_gspphot": [r"^logg$", r"^logg[-_]?gspphot$"],
    "radius_gspphot": [r"^rad$", r"^rad(ius)?[-_]?gspphot$"],
}
STAGE2_ROLE_EXTRA = {
    "t_start": [r"^begin$", r"^beg$", r"^tbeg$", r"^tbegin$"],
}


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass
class Stage2Params:
    tiers: tuple[str, ...] = ("interest", "watch")
    missions: tuple[str, ...] = ("kepler", "tess")
    max_stars: int = 40
    max_flares_per_star: int = 6
    # wall clocks
    budget_s: float = 9000.0
    per_star_budget_s: float = 1500.0
    retries: int = 2
    retry_pause_s: float = 5.0
    max_products: int = 4
    download_dir: str | None = None
    quality_bitmask: str = "default"
    # flare finder
    window_days: float = 0.5
    n_sigma: float = 3.0
    min_consecutive: int = 2
    peak_sigma: float = 4.0
    extend_sigma: float = 1.0
    t_flare_k: float = 9000.0
    # centroid
    baseline_days: float = 1.0
    exclude_cadences: int = 3
    core_fraction: float = 0.2
    sys_floor_px: float = 0.1
    n_sigma_centroid: float = 3.0
    min_snr: float = 3.0
    unresolved_px: float = 0.5
    trend_order: int = 1
    # census
    census_radius_arcsec: float = 12.0
    pixel_radius_arcsec: float = 4.0
    target_match_arcsec: float = 2.0
    max_neighbour_amplitude: float = 20.0
    prf_sigma_px_kepler: float = KEPLER_PRF_SIGMA_PX
    prf_sigma_px_tess: float = TESS_PRF_SIGMA_PX
    # parameters
    param_tables: dict = field(default_factory=lambda: {
        "kepler": ["J/AJ/159/280/table2", "J/AJ/159/280/table1", "V/133/kic"],
        "tess": ["IV/39/tic82", "IV/38/tic"]})
    flame_table: str = "I/355/paramp"
    gaia_table: str = "I/355/gaiadr3"
    # the ceiling
    b_gauss: float = B_DEFAULT_G
    geometric_factor: float = DEFAULT_GEOMETRIC_FACTOR
    independent_gap_days: float = 0.5
    sph_to_range: float = SPH_TO_RANGE

    @classmethod
    def from_config(cls, conf: dict | None) -> Stage2Params:
        c = dict((conf or {}).get("stage2") or {})
        phys = (conf or {}).get("physics") or {}
        d = cls()
        for k in ("tiers", "missions"):
            if c.get(k):
                setattr(d, k, tuple(str(x) for x in c[k]))
        for k in d.__dataclass_fields__:
            if k in ("tiers", "missions", "param_tables"):
                continue
            if c.get(k) is not None:
                cur = getattr(d, k)
                setattr(d, k, type(cur)(c[k]) if cur is not None else c[k])
        if c.get("param_tables"):
            d.param_tables = {str(k): [str(t) for t in v] for k, v in c["param_tables"].items()}
        for k in ("b_gauss", "geometric_factor", "independent_gap_days"):
            if phys.get(k) is not None:
                setattr(d, k, float(phys[k]))
        return d


@dataclass
class Deadline:
    budget_s: float | None = None
    started: float = field(default_factory=_time.monotonic)

    def expired(self) -> bool:
        return self.budget_s is not None and (_time.monotonic() - self.started) > self.budget_s

    def remaining(self) -> float:
        return float("inf") if self.budget_s is None else max(
            0.0, self.budget_s - (_time.monotonic() - self.started))

    def elapsed(self) -> float:
        return _time.monotonic() - self.started


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _finite(v) -> bool:
    return np.isfinite(_f(v))


# ---------------------------------------------------------------------------
# the shortlist
# ---------------------------------------------------------------------------
def load_shortlist(arc_dir: Path, *, params: Stage2Params) -> list[dict]:
    """Interest / candidate stars first, then watch, from ``candidates.json``."""
    p = Path(arc_dir) / "candidates.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return []
    order = {"candidate": 0, "interest": 1, "watch": 2}
    rows = list(d.get("candidates") or []) + list(d.get("watch") or [])
    rows = [r for r in rows if str(r.get("tier")) in set(params.tiers)
            and str(r.get("mission", "")).lower() in set(params.missions)]
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (order.get(str(r.get("tier")), 9),
                                         -_f(r.get("xi_conservative_max")))):
        if r.get("star_key") in seen:
            continue
        seen.add(r.get("star_key"))
        out.append(r)
    return out[:int(params.max_stars)]


# ---------------------------------------------------------------------------
# catalogue rows for one star
# ---------------------------------------------------------------------------
def _stage2_roles(columns, kind: str = "flares") -> dict:
    roles = resolve_arc_columns(columns, kind)
    if "t_start" not in roles:
        extra = resolve_columns(columns, {"t_start": STAGE2_ROLE_EXTRA["t_start"]})
        roles.update(extra)
    return roles


def fetch_flare_rows(star_id: str, spec: dict, *, query_fn=None, log: AcquisitionLog | None = None,
                     table: str | None = None) -> tuple[list[dict], str, dict]:
    """Every catalogue row for ``star_id`` from the flare table stage 1 used,
    with ``t_start`` / ``t_end`` / ``t_peak`` in the mission-native system,
    ``energy_erg`` and ``segment`` (quarter / sector).  Returns
    ``(rows, status, info)``."""
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog(prefix="arc/stage2")
    t = table or spec.get("table") or spec.get("preferred")
    mission = str(spec.get("mission", "kepler"))
    info = {"table": t, "roles": {}, "time_system": None}
    try:
        cols = table_columns(t, query_fn=query_fn)
    except Exception as exc:                              # noqa: BLE001
        log.record(f"flare_rows_{star_id}", f"columns of {t}", error=repr(exc))
        return [], STATUS_FAILED, info
    roles = _stage2_roles(cols, "flares")
    info["roles"] = roles
    if "star_id" not in roles or not ({"energy", "energy_max", "equiv_duration"} & set(roles)):
        log.record(f"flare_rows_{star_id}", f"roles of {t}", error=f"roles unresolved: {roles}")
        return [], STATUS_FAILED, info
    keep = [r for r in ("star_id", "t_peak", "t_start", "t_end", "energy", "energy_max",
                        "equiv_duration", "sector", "flag") if r in roles]
    sel = ", ".join(f'"{roles[r]}"' for r in keep)
    sid = str(star_id).strip()
    val = sid if sid.isdigit() else f"'{sid}'"
    adql = f'SELECT {sel} FROM "{t}" WHERE "{roles["star_id"]}" = {val}'
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record(f"flare_rows_{star_id}", adql, error=repr(exc))
        return [], STATUS_FAILED, info
    n = int(len(df)) if df is not None else 0
    log.record(f"flare_rows_{star_id}", adql, rows=n)
    if not n:
        return [], STATUS_ZERO, info
    out = pd.DataFrame({r: df.iloc[:, j] for j, r in enumerate(keep)})
    for c in ("t_peak", "t_start", "t_end", "energy", "energy_max", "equiv_duration"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "t_peak" not in out.columns and "t_start" in out.columns:
        out["t_peak"] = out["t_start"]
    tcol = next((c for c in ("t_peak", "t_start", "t_end") if c in out.columns), None)
    if tcol is not None:
        guess = guess_time_system(out[tcol].to_numpy(dtype=float), mission)
        out, native = normalise_time_system(out, guess, mission)
        info["time_system"] = native
        info["time_system_guess"] = guess
    espec = dict(spec.get("energy") or {})
    ecol = next((c for c in ("energy", "energy_max", "equiv_duration") if c in out.columns), None)
    kind = "equivalent_duration" if ecol == "equiv_duration" else str(espec.get("kind", "bolometric"))
    e_bol, einfo = energy_to_bolometric(out[ecol].to_numpy(dtype=float), kind=kind,
                                        factor=float(espec.get("factor", 1.0)),
                                        log10=espec.get("log10", "auto"),
                                        radius_rsun=np.ones(len(out)),
                                        t_star_k=np.full(len(out), 5772.0))
    info["energy"] = einfo
    rows = []
    for i in range(len(out)):
        r = {"energy_erg": float(e_bol[i]), "energy_raw": float(out[ecol].iloc[i]),
             "segment": (int(_f(out["sector"].iloc[i])) if "sector" in out.columns
                         and _finite(out["sector"].iloc[i]) else None)}
        for c in ("t_peak", "t_start", "t_end"):
            r[c] = float(out[c].iloc[i]) if c in out.columns and _finite(out[c].iloc[i]) \
                else float("nan")
        r["flag"] = str(out["flag"].iloc[i]) if "flag" in out.columns else ""
        rows.append(r)
    return rows, STATUS_OK, info


# ---------------------------------------------------------------------------
# stellar parameters
# ---------------------------------------------------------------------------
def fetch_gaia_flame(gaia_source: str | None, *, table: str = "I/355/paramp", query_fn=None,
                     log: AcquisitionLog | None = None) -> tuple[dict, str]:
    """Gaia DR3 FLAME radius / mass / luminosity / age (and the GSP-Phot Teff /
    logg / radius) for one source id from the astrophysical-parameters table,
    with the columns discovered at runtime."""
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog(prefix="arc/stage2")
    out: dict = {"table": table}
    if not gaia_source or not str(gaia_source).strip().isdigit():
        return out, STATUS_ZERO
    try:
        cols = table_columns(table, query_fn=query_fn)
    except Exception as exc:                              # noqa: BLE001
        log.record("gaia_flame", f"columns of {table}", error=repr(exc))
        return out, STATUS_FAILED
    roles = resolve_columns(cols, FLAME_PATTERNS)
    out["roles"] = roles
    if "gaia_id" not in roles:
        log.record("gaia_flame", f"roles of {table}", error=f"no source id column in {cols[:20]}")
        return out, STATUS_FAILED
    keep = [r for r in roles if r != "gaia_id"]
    if not keep:
        return out, STATUS_ZERO
    sel = ", ".join(f'"{roles[r]}"' for r in ["gaia_id"] + keep)
    adql = f'SELECT {sel} FROM "{table}" WHERE "{roles["gaia_id"]}" = {int(gaia_source)}'
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record("gaia_flame", adql, error=repr(exc))
        return out, STATUS_FAILED
    n = int(len(df)) if df is not None else 0
    log.record("gaia_flame", adql, rows=n)
    if not n:
        return out, STATUS_ZERO
    for j, r in enumerate(["gaia_id"] + keep):
        v = df.iloc[0, j]
        out[r] = str(v) if r == "gaia_id" else _f(v)
    return out, STATUS_OK


def stellar_parameters(star_id: str, mission: str, *, params: Stage2Params, query_fn=None,
                       log: AcquisitionLog | None = None, gaia_source: str | None = None,
                       stage1: dict | None = None) -> dict:
    """Teff / radius / logg / mass with their sources: Berger+2020 first, the
    KIC / TIC next, Gaia FLAME beside them; stage 1's values as the fallback,
    flagged.  ``measured`` says whether anything better than stage 1 arrived."""
    log = log or AcquisitionLog(prefix="arc/stage2")
    rec = {"teff_k": float("nan"), "radius_rsun": float("nan"), "logg": float("nan"),
           "mass_msun": float("nan"), "teff_source": None, "radius_source": None,
           "logg_source": None, "mass_source": None, "ra": float("nan"), "dec": float("nan"),
           "evol": None, "flag": "", "gaia_ruwe": float("nan"), "measured": False,
           "tables": [], "flame": {}, "flame_status": None}
    pos, prec = fetch_star_params_by_id([str(star_id)], mission, query_fn=query_fn, log=log,
                                        tables=params.param_tables)
    rec["tables"] = prec
    if len(pos):
        h = pos.iloc[0]
        for col, key in (("teff", "teff_k"), ("radius", "radius_rsun"), ("logg", "logg"),
                         ("mass", "mass_msun")):
            v = _f(h.get(col))
            if np.isfinite(v) and v > 0:
                rec[key] = v
                rec[key.split("_")[0] + "_source"] = str(h.get(f"{col}_source")
                                                         or h.get("params_source") or "")
        rec["ra"], rec["dec"] = _f(h.get("ra")), _f(h.get("dec"))
        rec["evol"] = None if pd.isna(h.get("evol")) else str(h.get("evol"))
        rec["flag"] = "" if pd.isna(h.get("flag")) else str(h.get("flag"))
        rec["gaia_ruwe"] = _f(h.get("ruwe"))
        if not gaia_source and not pd.isna(h.get("gaia_id")):
            gaia_source = str(h.get("gaia_id")).split(".")[0]
    flame, fstat = fetch_gaia_flame(gaia_source, table=params.flame_table, query_fn=query_fn,
                                    log=log)
    rec["flame"], rec["flame_status"] = flame, fstat
    if fstat == STATUS_OK:
        if not np.isfinite(rec["radius_rsun"]) and _finite(flame.get("radius_flame")):
            rec["radius_rsun"], rec["radius_source"] = _f(flame["radius_flame"]), "gaia_flame"
        if not np.isfinite(rec["mass_msun"]) and _finite(flame.get("mass_flame")):
            rec["mass_msun"], rec["mass_source"] = _f(flame["mass_flame"]), "gaia_flame"
        if not np.isfinite(rec["teff_k"]) and _finite(flame.get("teff_gspphot")):
            rec["teff_k"], rec["teff_source"] = _f(flame["teff_gspphot"]), "gaia_gspphot"
        if not np.isfinite(rec["logg"]) and _finite(flame.get("logg_gspphot")):
            rec["logg"], rec["logg_source"] = _f(flame["logg_gspphot"]), "gaia_gspphot"
    rec["measured"] = bool(np.isfinite(rec["teff_k"]) and np.isfinite(rec["radius_rsun"]))
    s1 = stage1 or {}
    if not np.isfinite(rec["teff_k"]) and _finite(s1.get("teff_k")):
        rec["teff_k"], rec["teff_source"] = _f(s1["teff_k"]), f"stage1:{s1.get('teff_source')}"
    if not np.isfinite(rec["radius_rsun"]) and _finite(s1.get("radius_rsun")):
        rec["radius_rsun"], rec["radius_source"] = _f(s1["radius_rsun"]), "stage1"
    if not np.isfinite(rec["logg"]) and _finite(s1.get("logg")):
        rec["logg"], rec["logg_source"] = _f(s1["logg"]), "stage1"
    return rec


# ---------------------------------------------------------------------------
# the Gaia census
# ---------------------------------------------------------------------------
def _sep_pa(ra0, dec0, ra, dec) -> tuple[float, float]:
    p0, p1 = math.radians(dec0), math.radians(dec)
    dl = math.radians(ra - ra0)
    h = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(dl / 2) ** 2
    sep = math.degrees(2 * math.asin(min(1.0, math.sqrt(h)))) * 3600.0
    y = math.sin(dl) * math.cos(p1)
    x = math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl)
    return sep, math.degrees(math.atan2(y, x)) % 360.0


def gaia_sources(ra: float, dec: float, *, params: Stage2Params, cone_fn=None,
                 log: AcquisitionLog | None = None, key: str = "") -> tuple[list[dict], int, str]:
    """Every Gaia DR3 source within the census radius as
    ``{id, ra, dec, gmag, sep_arcsec, pa_deg, ruwe, plx}``, plus the index of
    the target (nearest within ``target_match_arcsec``; -1 when none)."""
    cone_fn = cone_fn or _vizier_cone
    log = log or AcquisitionLog(prefix="arc/stage2")
    label = f"gaia_census_{key or 'target'}"
    if not (np.isfinite(ra) and np.isfinite(dec)):
        log.record(label, "no position", error="target has no ra/dec")
        return [], -1, STATUS_FAILED
    try:
        df = cone_fn(params.gaia_table, float(ra), float(dec), float(params.census_radius_arcsec))
    except Exception as exc:                              # noqa: BLE001
        log.record(label, f"cone {params.gaia_table} r={params.census_radius_arcsec}",
                   error=repr(exc)[:300])
        return [], -1, STATUS_FAILED
    n = int(len(df)) if df is not None else 0
    log.record(label, f"cone {params.gaia_table} r={params.census_radius_arcsec}", rows=n)
    if not n:
        return [], -1, STATUS_ZERO
    roles = resolve_columns(df.columns, {k: ROLE_PATTERNS[k] for k in
                                         ("ra", "dec", "gmag", "gaia_id", "ruwe", "plx")})
    if "ra" not in roles or "dec" not in roles:
        return [], -1, STATUS_FAILED
    out = []
    for i in range(n):
        r_, d_ = _f(df[roles["ra"]].iloc[i]), _f(df[roles["dec"]].iloc[i])
        if not (np.isfinite(r_) and np.isfinite(d_)):
            continue
        sep, pa = _sep_pa(float(ra), float(dec), r_, d_)
        out.append({"id": (str(df[roles["gaia_id"]].iloc[i]).split(".")[0]
                           if "gaia_id" in roles else str(i)),
                    "ra": r_, "dec": d_,
                    "gmag": _f(df[roles["gmag"]].iloc[i]) if "gmag" in roles else float("nan"),
                    "sep_arcsec": sep, "pa_deg": pa,
                    "ruwe": _f(df[roles["ruwe"]].iloc[i]) if "ruwe" in roles else float("nan"),
                    "plx": _f(df[roles["plx"]].iloc[i]) if "plx" in roles else float("nan")})
    out.sort(key=lambda s: s["sep_arcsec"])
    tidx = 0 if out and out[0]["sep_arcsec"] <= float(params.target_match_arcsec) else -1
    return out, tidx, STATUS_OK


# ---------------------------------------------------------------------------
# MAST: light curves and target pixel files (runner only; injectable)
# ---------------------------------------------------------------------------
def mast_probe() -> dict:
    rep = {"lightkurve": {"importable": False}, "astroquery_mast": {"importable": False},
           "astropy_io_fits": {"importable": False}, "route_preferred": None}
    try:
        import lightkurve as lk  # noqa: PLC0415

        rep["lightkurve"] = {"importable": True, "version": str(getattr(lk, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["lightkurve"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        import astroquery  # noqa: PLC0415
        from astroquery.mast import Observations  # noqa: PLC0415, F401

        rep["astroquery_mast"] = {"importable": True,
                                  "version": str(getattr(astroquery, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["astroquery_mast"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        from astropy.io import fits  # noqa: PLC0415, F401

        rep["astropy_io_fits"] = {"importable": True}
    except Exception as exc:                              # noqa: BLE001
        rep["astropy_io_fits"] = {"importable": False, "error": repr(exc)[:300]}
    if rep["lightkurve"]["importable"]:
        rep["route_preferred"] = ROUTE_LIGHTKURVE
    elif rep["astroquery_mast"]["importable"] and rep["astropy_io_fits"]["importable"]:
        rep["route_preferred"] = ROUTE_MAST_FITS
    return rep


def _time_offset(mission: str) -> float:
    return BTJD_OFFSET if str(mission).lower().startswith("tess") else BKJD_OFFSET


def _target_name(star_id, mission: str) -> str:
    return (f"TIC {int(star_id)}" if str(mission).lower().startswith("tess")
            else f"KIC {int(star_id)}")


def _wcs_pix_fn(wcs):
    if wcs is None:
        return None

    def _fn(ra, dec):
        px = wcs.all_world2pix([[float(ra), float(dec)]], 0)
        return float(px[0][0]), float(px[0][1])

    try:
        _fn(0.0, 0.0)
    except Exception:                                     # noqa: BLE001
        return None
    return _fn


def _pixel_record_from_lightkurve(tpf, mission: str, *, cutout: bool = False,
                                  ra: float = float("nan"), dec: float = float("nan")) -> dict:
    off = _time_offset(mission)
    try:
        t = np.asarray(tpf.time.jd, dtype=float) - off
    except Exception:                                     # noqa: BLE001
        t = np.asarray(tpf.time.value, dtype=float)
    cube = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
    err = np.asarray(getattr(getattr(tpf, "flux_err", None), "value",
                             np.full(cube.shape, np.nan)), dtype=float)
    meta = dict(getattr(tpf, "meta", {}) or {})
    exptime = meta.get("TIMEDEL")
    exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
    if not np.isfinite(exptime_s) and t.size > 2:
        exptime_s = float(np.median(np.diff(np.sort(t)))) * 86400.0
    seg = meta.get("QUARTER") if not str(mission).lower().startswith("tess") else meta.get("SECTOR")
    rec = {"mission": str(mission).lower(), "segment": int(seg) if seg is not None else None,
           "time": t, "flux": cube, "flux_err": err,
           "quality": np.asarray(getattr(tpf, "quality", np.zeros(t.size)), dtype=int),
           "exptime_s": exptime_s, "column": getattr(tpf, "column", None),
           "row": getattr(tpf, "row", None), "author": str(meta.get("AUTHOR") or
                                                            meta.get("ORIGIN") or "unknown"),
           "channel": meta.get("CHANNEL"), "n_cadences": int(t.size), "cutout": bool(cutout),
           "pixel_scale_arcsec": (TESS_PIXEL_ARCSEC if str(mission).lower().startswith("tess")
                                  else KEPLER_PIXEL_ARCSEC)}
    ap = getattr(tpf, "pipeline_mask", None)
    if ap is not None and np.asarray(ap).any() and not cutout:
        rec["aperture"] = np.asarray(ap, dtype=bool)
        rec["aperture_source"] = "pipeline"
    fn = _wcs_pix_fn(getattr(tpf, "wcs", None))
    if fn is not None:
        rec["pix_fn"] = fn
        try:
            rec["target_pixel"] = fn(ra, dec) if np.isfinite(ra) and np.isfinite(dec) else None
        except Exception:                                 # noqa: BLE001
            rec["target_pixel"] = None
    if "aperture" not in rec:
        ny, nx = cube.shape[1:]
        cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0)
        tp = rec.get("target_pixel")
        if tp is not None and all(np.isfinite(tp)):
            cx, cy = tp
        yy, xx = np.mgrid[0:ny, 0:nx]
        rec["aperture"] = np.hypot(xx - cx, yy - cy) <= 1.5
        rec["aperture_source"] = "synthetic_disc_1.5px"
    return rec


def lightkurve_pixels_fn(star_id, *, mission: str, segments=(), max_products: int = 4,
                         download_dir: str | None = None, ra: float = float("nan"),
                         dec: float = float("nan"), quality_bitmask: str = "default",
                         **_kw) -> list[dict]:
    """Target pixel files for the wanted quarters / sectors through ``lightkurve``."""
    import lightkurve as lk  # noqa: PLC0415

    name = _target_name(star_id, mission)
    tess = str(mission).lower().startswith("tess")
    out: list[dict] = []
    for seg in (list(segments) or [None]):
        cutout = False
        if tess:
            kw = {"sector": int(seg)} if seg is not None else {}
            sr = lk.search_targetpixelfile(name, mission="TESS", author="SPOC", **kw)
            if sr is None or len(sr) == 0:
                sr = lk.search_tesscut(name, **kw)
                cutout = True
        else:
            kw = {"quarter": int(seg)} if seg is not None else {}
            sr = lk.search_targetpixelfile(name, mission="Kepler", author="Kepler",
                                           cadence="long", **kw)
        if sr is None or len(sr) == 0:
            continue
        for i in range(min(len(sr), int(max_products))):
            try:
                if cutout:
                    tpf = sr[i].download(download_dir=download_dir, cutout_size=11,
                                         quality_bitmask=quality_bitmask)
                else:
                    tpf = sr[i].download(download_dir=download_dir,
                                         quality_bitmask=quality_bitmask)
            except Exception:                             # noqa: BLE001
                continue
            if tpf is None:
                continue
            out.append(_pixel_record_from_lightkurve(tpf, mission, cutout=cutout, ra=ra, dec=dec))
    return out


def read_pixel_fits(path, *, mission: str = "kepler", ra: float = float("nan"),
                    dec: float = float("nan")) -> dict | None:
    """A Kepler / TESS target pixel FITS product -> the pixel record."""
    from astropy.io import fits  # noqa: PLC0415

    off = _time_offset(mission)
    with fits.open(str(path), memmap=False) as hdul:
        hdu = None
        for h in hdul:
            try:
                names = [str(n).upper() for n in h.columns.names]
            except AttributeError:
                continue
            if "TIME" in names and "FLUX" in names:
                hdu = h
                break
        if hdu is None:
            return None
        data, hdr = hdu.data, hdu.header
        cols = [str(n).upper() for n in hdu.columns.names]
        bjdrefi = float(hdr.get("BJDREFI", hdul[0].header.get("BJDREFI", off)) or off)
        bjdreff = float(hdr.get("BJDREFF", hdul[0].header.get("BJDREFF", 0.0)) or 0.0)
        t = np.asarray(data["TIME"], dtype=float) + (bjdrefi + bjdreff) - off
        cube = np.asarray(data["FLUX"], dtype=float)
        err = np.asarray(data["FLUX_ERR"], dtype=float) if "FLUX_ERR" in cols else None
        q = np.asarray(data["QUALITY"], dtype=int) if "QUALITY" in cols else np.zeros(t.size, int)
        exptime = hdr.get("TIMEDEL")
        exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
        p0 = hdul[0].header
        seg = p0.get("SECTOR") if str(mission).lower().startswith("tess") else p0.get("QUARTER")
        rec = {"mission": str(mission).lower(), "segment": int(seg) if seg is not None else None,
               "time": t, "flux": cube, "flux_err": err, "quality": q, "exptime_s": exptime_s,
               "column": hdr.get("1CRV5P") or hdr.get("CRVAL1P"),
               "row": hdr.get("2CRV5P") or hdr.get("CRVAL2P"),
               "author": str(p0.get("ORIGIN", "") or "unknown"), "channel": p0.get("CHANNEL"),
               "n_cadences": int(t.size), "cutout": False,
               "pixel_scale_arcsec": (TESS_PIXEL_ARCSEC if str(mission).lower().startswith("tess")
                                      else KEPLER_PIXEL_ARCSEC)}
        try:
            ap_img = np.asarray(hdul["APERTURE"].data)
            rec["aperture"] = (ap_img & 2) > 0
            rec["aperture_source"] = "pipeline"
        except Exception:                                 # noqa: BLE001
            ny, nx = cube.shape[1:]
            yy, xx = np.mgrid[0:ny, 0:nx]
            rec["aperture"] = np.hypot(xx - (nx - 1) / 2, yy - (ny - 1) / 2) <= 1.5
            rec["aperture_source"] = "synthetic_disc_1.5px"
        try:
            from astropy.wcs import WCS  # noqa: PLC0415

            fn = _wcs_pix_fn(WCS(hdul["APERTURE"].header))
            if fn is not None:
                rec["pix_fn"] = fn
                rec["target_pixel"] = fn(ra, dec) if np.isfinite(ra) and np.isfinite(dec) else None
        except Exception:                                 # noqa: BLE001
            pass
        return rec


def mast_fits_pixels_fn(star_id, *, mission: str, segments=(), max_products: int = 4,
                        download_dir: str | None = None, ra: float = float("nan"),
                        dec: float = float("nan"), **_kw) -> list[dict]:
    """Target pixel files through ``astroquery.mast`` + FITS (the fallback route).
    Kepler products are ``TPL`` (long cadence); the quarter is only known from
    the header, so every product is read and the wanted ones kept."""
    from astroquery.mast import Observations  # noqa: PLC0415

    tess = str(mission).lower().startswith("tess")
    coll = "TESS" if tess else "Kepler"
    forms = [str(int(star_id))] if tess else [f"kplr{int(star_id):09d}", str(int(star_id))]
    obs = None
    for form in forms:
        try:
            cand = Observations.query_criteria(obs_collection=coll, dataproduct_type="timeseries",
                                               target_name=form)
        except Exception:                                 # noqa: BLE001
            continue
        if cand is not None and len(cand):
            obs = cand
            break
    if obs is None or len(obs) == 0:
        return []
    prod = Observations.get_product_list(obs)
    if prod is None or len(prod) == 0:
        return []
    p = prod.to_pandas()
    if "productSubGroupDescription" in p.columns:
        want = {"TP"} if tess else {"TPL"}
        sel = p["productSubGroupDescription"].astype(str).str.upper().isin(want)
        if sel.any():
            p = p[sel]
    root = Path(download_dir or ".") / "mast_arc_tpf"
    root.mkdir(parents=True, exist_ok=True)
    wanted = {int(s) for s in segments} if segments else None
    out: list[dict] = []
    for _, r in p.iterrows():
        if len(out) >= int(max_products):
            break
        uri = str(r.get("dataURI") or "")
        if not uri:
            continue
        local = root / str(r.get("productFilename") or Path(uri).name)
        try:
            status, _m, _u = Observations.download_file(uri, local_path=str(local), cache=False)
            if str(status).upper() != "COMPLETE" or not local.exists():
                continue
            rec = read_pixel_fits(local, mission=mission, ra=ra, dec=dec)
        except Exception:                                 # noqa: BLE001
            continue
        if rec is None:
            continue
        if wanted is not None and (rec.get("segment") is None or int(rec["segment"]) not in wanted):
            continue
        out.append(rec)
    return out


def default_pixels_fn(star_id, **kw) -> list[dict]:
    try:
        import lightkurve  # noqa: PLC0415, F401

        return lightkurve_pixels_fn(star_id, **kw)
    except ImportError:
        return mast_fits_pixels_fn(star_id, **kw)


def lightkurve_lc_fn(star_id, *, mission: str, segments=(), max_products: int = 4,
                     download_dir: str | None = None, quality_bitmask: str = "default",
                     **_kw) -> list[dict]:
    """PDCSAP (and SAP) light curves for the wanted quarters / sectors."""
    import lightkurve as lk  # noqa: PLC0415

    name = _target_name(star_id, mission)
    tess = str(mission).lower().startswith("tess")
    off = _time_offset(mission)
    out: list[dict] = []
    for seg in (list(segments) or [None]):
        if tess:
            kw = {"sector": int(seg)} if seg is not None else {}
            sr = lk.search_lightcurve(name, mission="TESS", author="SPOC", **kw)
        else:
            kw = {"quarter": int(seg)} if seg is not None else {}
            sr = lk.search_lightcurve(name, mission="Kepler", author="Kepler", cadence="long", **kw)
        if sr is None or len(sr) == 0:
            continue
        for i in range(min(len(sr), int(max_products))):
            try:
                lc = sr[i].download(download_dir=download_dir, quality_bitmask=quality_bitmask)
            except Exception:                             # noqa: BLE001
                continue
            if lc is None:
                continue
            try:
                t = np.asarray(lc.time.jd, dtype=float) - off
            except Exception:                             # noqa: BLE001
                t = np.asarray(lc.time.value, dtype=float)
            meta = dict(getattr(lc, "meta", {}) or {})
            rec = {"mission": str(mission).lower(),
                   "segment": int(meta.get("SECTOR") if tess else meta.get("QUARTER"))
                   if (meta.get("SECTOR") if tess else meta.get("QUARTER")) is not None else None,
                   "time": t, "quality": np.asarray(getattr(lc, "quality", np.zeros(t.size)),
                                                    dtype=int),
                   "flux_columns": {}, "author": str(meta.get("AUTHOR") or "unknown")}
            for col in ("pdcsap_flux", "sap_flux"):
                try:
                    v = np.asarray(lc[col].value, dtype=float)
                    e = np.asarray(lc[col + "_err"].value, dtype=float) if col + "_err" in \
                        lc.columns else np.full(v.shape, np.nan)
                    rec["flux_columns"][col] = (v, e)
                except Exception:                         # noqa: BLE001
                    continue
            if not rec["flux_columns"]:
                v = np.asarray(getattr(lc.flux, "value", lc.flux), dtype=float)
                rec["flux_columns"]["flux"] = (v, np.full(v.shape, np.nan))
            out.append(rec)
    return out


def lc_from_pixels(rec: dict) -> dict:
    """An aperture-sum light curve from a pixel record (the fallback when the
    light-curve product did not arrive)."""
    cube = np.asarray(rec["flux"], dtype=float)
    ap = np.asarray(rec["aperture"], dtype=bool)
    w = np.where(ap[None] & np.isfinite(cube), cube, 0.0)
    f = w.sum(axis=(1, 2))
    err = rec.get("flux_err")
    if err is not None and np.asarray(err).shape == cube.shape:
        e = np.sqrt(np.where(ap[None] & np.isfinite(err), np.asarray(err) ** 2, 0.0).sum(axis=(1, 2)))
    else:
        e = np.full(f.shape, np.nan)
    return {"mission": rec.get("mission"), "segment": rec.get("segment"), "time": rec["time"],
            "quality": rec.get("quality"), "flux_columns": {"aperture_sum": (f, e)},
            "author": "aperture_sum_of_pixels"}


def _fetch_products(star_id, *, mission, segments, fn, default_fn, params: Stage2Params,
                    log: AcquisitionLog, deadline: Deadline | None, label: str, ra, dec
                    ) -> tuple[list[dict], str, str]:
    if deadline is not None and deadline.expired():
        log.record(label, f"{mission} {star_id} segments={list(segments)}",
                   error="budget_exhausted_before_request")
        return [], STATUS_FAILED, "budget_exhausted"
    route = "injected" if fn is not None else (mast_probe().get("route_preferred") or "")
    f = fn or default_fn
    last = ""
    for attempt in range(max(int(params.retries), 1)):
        if deadline is not None and deadline.expired():
            last = "budget_exhausted"
            break
        if attempt and float(params.retry_pause_s) > 0:
            _time.sleep(min(float(params.retry_pause_s) * attempt,
                            deadline.remaining() if deadline is not None else 60.0))
        try:
            recs = f(star_id, mission=mission, segments=tuple(segments),
                     max_products=int(params.max_products), download_dir=params.download_dir,
                     ra=ra, dec=dec, quality_bitmask=params.quality_bitmask)
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)[:400]
            continue
        recs = [r for r in (recs or []) if r is not None]
        if not recs:
            log.record(label, f"{mission} {star_id} segments={list(segments)}", rows=0,
                       extra={"route": route, "attempt": attempt + 1})
            return [], STATUS_ZERO, route
        log.record(label, f"{mission} {star_id} segments={list(segments)}",
                   rows=int(sum(int(np.asarray(r.get("time")).size) for r in recs)),
                   extra={"route": route, "n_products": len(recs), "attempt": attempt + 1})
        return recs, STATUS_OK, route
    log.record(label, f"{mission} {star_id} segments={list(segments)}", error=last or "unknown")
    return [], STATUS_FAILED, route


# ---------------------------------------------------------------------------
# one star
# ---------------------------------------------------------------------------
def _flares_to_test(rows: list[dict], entry: dict, params: Stage2Params) -> list[dict]:
    """The rows above the conservative ceiling (or, for a watch star, above
    the nominal one), largest first, capped."""
    e_con, e_nom = _f(entry.get("e_mag_conservative_erg")), _f(entry.get("e_mag_nominal_erg"))
    above = [r for r in rows if np.isfinite(e_con) and r["energy_erg"] > e_con]
    tier = "conservative"
    if not above:
        above = [r for r in rows if np.isfinite(e_nom) and r["energy_erg"] > e_nom]
        tier = "nominal"
    above.sort(key=lambda r: -r["energy_erg"])
    for r in above:
        r["above"] = tier
        r["xi_conservative_stage1"] = (math.log10(r["energy_erg"] / e_con)
                                       if np.isfinite(e_con) and e_con > 0 else float("nan"))
    return above[:int(params.max_flares_per_star)]


def _best_lc(lcs: list[dict], segment) -> tuple[dict | None, str]:
    for rec in lcs:
        if segment is not None and rec.get("segment") is not None and int(rec["segment"]) != int(segment):
            continue
        cols = rec.get("flux_columns") or {}
        for c in ("pdcsap_flux", "sap_flux", "flux", "aperture_sum"):
            if c in cols:
                return rec, c
    return None, ""


def _xi_block(entry: dict, rows: list[dict], prm: dict, amp_q: dict, remeasured: list[float],
              params: Stage2Params) -> dict:
    """xi recomputed on the measured parameters with (a) the catalogue
    amplitude (Sph rescaled to a range) and (b) the quarter's own amplitude;
    the headline is the larger amplitude, the conservative choice."""
    teff, rad = _f(prm.get("teff_k")), _f(prm.get("radius_rsun"))
    amp_cat = _f(entry.get("amplitude_frac"))
    src = str(entry.get("amplitude_source") or "")
    scale = 1.0
    if src == "santos2021" and not bool(entry.get("amplitude_scaled", False)):
        scale = float(params.sph_to_range)
    amp_cat_s = amp_cat * scale if np.isfinite(amp_cat) else float("nan")
    amp_rvar = _f(amp_q.get("rvar"))
    e = np.array([r["energy_erg"] for r in rows], dtype=float)
    t = np.array([r.get("t_peak", np.nan) for r in rows], dtype=float)
    t = t if np.isfinite(t).any() else None
    kw = {"b_gauss": float(params.b_gauss), "geometric_factor": float(params.geometric_factor),
          "independent_gap_days": float(params.independent_gap_days)}

    def _run(amp, energies, times):
        if not (np.isfinite(amp) and amp > 0 and len(energies)):
            return {"assessable": False, "xi_conservative_max": float("nan"),
                    "xi_nominal_max": float("nan"), "n_above_conservative": 0}
        rec = star_ceiling(np.asarray(energies, dtype=float), times, float(amp), rad, teff, **kw)
        return {k: rec.get(k) for k in ("assessable", "xi_conservative_max", "xi_nominal_max",
                                        "n_above_conservative", "n_above_nominal",
                                        "e_mag_conservative_erg", "e_mag_nominal_erg",
                                        "a_spot_conservative_cm2", "n_independent")}

    amps = {"catalogue_amplitude": amp_cat_s, "quarter_rvar": amp_rvar}
    fin = {k: v for k, v in amps.items() if np.isfinite(v) and v > 0}
    amp_max = max(fin.values()) if fin else float("nan")
    amp_max_src = max(fin, key=fin.get) if fin else None
    block = {
        "xi_conservative_stage1": _f(entry.get("xi_conservative_max")),
        "teff_k": teff, "radius_rsun": rad, "params_measured": bool(prm.get("measured")),
        "amplitude_catalogue_frac": amp_cat, "amplitude_catalogue_source": src,
        "amplitude_catalogue_scale_applied": scale, "amplitude_catalogue_as_range": amp_cat_s,
        "amplitude_quarter_rvar": amp_rvar, "amplitude_quarter_sph": _f(amp_q.get("sph")),
        "amplitude_used": amp_max, "amplitude_used_source": amp_max_src,
        "n_catalogue_flares": int(len(rows)),
        "with_catalogue_amplitude": _run(amp_cat_s, e, t),
        "with_quarter_amplitude": _run(amp_rvar, e, t),
        "with_max_amplitude": _run(amp_max, e, t),
    }
    rm = [x for x in remeasured if np.isfinite(x) and x > 0]
    block["with_max_amplitude_remeasured_energy"] = _run(amp_max, rm, None) if rm else {
        "assessable": False, "xi_conservative_max": float("nan"), "n_above_conservative": 0}
    head = block["with_max_amplitude"]
    x = _f(head.get("xi_conservative_max"))
    block["xi_conservative_measured"] = x
    block["xi_status"] = ("above_ceiling" if np.isfinite(x) and x > 0 else
                          ("below_ceiling" if np.isfinite(x) else "unassessable"))
    return block


def analyse_star(entry: dict, *, conf: dict, params: Stage2Params, query_fn=None, cone_fn=None,
                 lc_fn=None, tpf_fn=None, log: AcquisitionLog | None = None,
                 deadline: Deadline | None = None) -> dict:
    """Everything stage 2 does for one shortlisted star.  Never raises on a
    missing product: every gap is a stated status."""
    log = log or AcquisitionLog(prefix="arc/stage2")
    own = Deadline(budget_s=float(params.per_star_budget_s))
    sid, mission = str(entry.get("star_id")), str(entry.get("mission", "kepler")).lower()
    key = str(entry.get("star_key") or f"{mission}:{sid}")
    cat = str(entry.get("catalogue") or "")
    spec = dict(((conf.get("catalogues") or {}).get(cat)) or {"mission": mission})
    ctx = entry.get("context") or {}
    rec: dict = {"star_key": key, "star_id": sid, "mission": mission, "catalogue": cat,
                 "tier": entry.get("tier"), "xi_conservative_stage1": _f(entry.get("xi_conservative_max")),
                 "generated_utc": _now(), "statuses": {}, "degraded": [], "flares": [],
                 "census": {}, "verdict": VERDICT_UNTESTABLE, "verdict_reason": "",
                 "route": {}}

    def _deadline() -> Deadline | None:
        # the tighter of the star's own clock and the stage clock
        if deadline is None:
            return own
        return own if own.remaining() < deadline.remaining() else deadline

    # --- 1. the catalogue rows ------------------------------------------------
    table = None
    try:
        acq = json.loads((Path(conf.get("_arc_dir", "results/arc")) / "acquire.json").read_text())
        table = ((acq.get("catalogues") or {}).get(cat) or {}).get("table")
    except Exception:                                     # noqa: BLE001
        table = None
    rows, rstat, rinfo = fetch_flare_rows(sid, spec, query_fn=query_fn, log=log, table=table)
    rec["statuses"]["flare_rows"] = rstat
    rec["flare_rows_info"] = rinfo
    if rstat != STATUS_OK:
        # fall back to stage 1's pull list (energies only)
        rows = [{"energy_erg": _f(p.get("e_flare_erg")), "t_peak": _f(p.get("t_peak")),
                 "t_start": float("nan"), "t_end": float("nan"),
                 "segment": p.get("quarter", p.get("sector"))}
                for p in (entry.get("stage2_pulls") or [])]
        rec["degraded"].append(f"flare_rows:{rstat}")
    tests = _flares_to_test(rows, entry, params)
    rec["n_catalogue_rows"], rec["n_flares_to_test"] = int(len(rows)), int(len(tests))
    segments = sorted({int(r["segment"]) for r in tests if r.get("segment") is not None})
    for r in tests:
        if r.get("segment") is None and mission == "kepler" and _finite(r.get("t_peak")):
            from .vet import kepler_quarter_of  # noqa: PLC0415

            q = kepler_quarter_of(r["t_peak"])
            if q is not None:
                r["segment"] = q
                segments = sorted(set(segments) | {q})
    rec["segments"] = segments

    # --- 2. stellar parameters --------------------------------------------------
    prm = stellar_parameters(sid, mission, params=params, query_fn=query_fn, log=log,
                             gaia_source=ctx.get("gaia_source"), stage1=entry)
    rec["params"] = prm
    rec["statuses"]["params"] = STATUS_OK if prm.get("measured") else STATUS_ZERO
    if not prm.get("measured"):
        rec["degraded"].append("stellar_params:not_measured")
    ra, dec = _f(prm.get("ra")), _f(prm.get("dec"))

    # --- 3. the Gaia census (positions) -------------------------------------------
    srcs, tidx, gstat = gaia_sources(ra, dec, params=params, cone_fn=cone_fn, log=log, key=sid)
    rec["statuses"]["gaia_census"] = gstat
    if gstat != STATUS_OK or tidx < 0:
        rec["degraded"].append(f"gaia_census:{gstat if gstat != STATUS_OK else 'target_not_matched'}")
    rec["n_gaia_sources"] = int(len(srcs))

    # --- 4. light curves and pixel files for the flare segments -----------------------
    lcs, lstat, lroute = ([], STATUS_ZERO, "") if not tests else _fetch_products(
        sid, mission=mission, segments=segments, fn=lc_fn, default_fn=lightkurve_lc_fn,
        params=params, log=log, deadline=_deadline(), label=f"lightcurve_{sid}", ra=ra, dec=dec)
    tpfs, tstat, troute = ([], STATUS_ZERO, "") if not tests else _fetch_products(
        sid, mission=mission, segments=segments, fn=tpf_fn, default_fn=default_pixels_fn,
        params=params, log=log, deadline=_deadline(), label=f"pixels_{sid}", ra=ra, dec=dec)
    rec["statuses"]["lightcurve"], rec["statuses"]["pixels"] = lstat, tstat
    rec["route"] = {"lightcurve": lroute, "pixels": troute}
    if not tests:
        rec["degraded"].append("no_flare_above_ceiling_to_test")
    if tests and lstat != STATUS_OK:
        rec["degraded"].append(f"lightcurve:{lstat}")
    if tests and tstat != STATUS_OK:
        rec["degraded"].append(f"pixels:{tstat}")
    if lstat != STATUS_OK and tpfs:
        lcs = [lc_from_pixels(p) for p in tpfs]
        rec["degraded"].append("lightcurve:aperture_sum_of_pixels_used")

    # --- 5. per segment: detect, match, energy, amplitude ---------------------------
    cad_default = (2.0 / 1440.0) if mission == "tess" else (29.4244 / 1440.0)
    prf_sigma = params.prf_sigma_px_tess if mission == "tess" else params.prf_sigma_px_kepler
    amp_by_seg: dict = {}
    teff, rad = _f(prm.get("teff_k")), _f(prm.get("radius_rsun"))
    flare_recs, remeasured = [], []
    census_rows: list[dict] = []
    for seg in (segments or [None]):
        seg_tests = [r for r in tests if r.get("segment") == seg or seg is None]
        lc, fcol = _best_lc(lcs, seg)
        if lc is None:
            for r in seg_tests:
                flare_recs.append({**{k: r.get(k) for k in ("energy_erg", "t_start", "t_end",
                                                             "t_peak", "segment", "above",
                                                             "xi_conservative_stage1")},
                                   "match_method": MATCH_NONE, "outcome": OUTCOME_UNTESTABLE,
                                   "reason": "no light curve for this segment"})
            continue
        flux, ferr = lc["flux_columns"][fcol]
        t = np.asarray(lc["time"], dtype=float)
        det, dinfo = detect_flares(t, flux, window_days=params.window_days, n_sigma=params.n_sigma,
                                   min_consecutive=params.min_consecutive,
                                   peak_sigma=params.peak_sigma, extend_sigma=params.extend_sigma,
                                   quality=lc.get("quality"))
        cad = dinfo.get("cadence_days") if np.isfinite(_f(dinfo.get("cadence_days"))) else cad_default
        fmask = np.zeros(t.shape, dtype=bool)
        order = np.argsort(t)
        for d in det:
            lo, hi = d["t_start"] - cad, d["t_end"] + cad
            fmask |= (t >= lo) & (t <= hi)
        amp_q = rotational_amplitude(t, flux, flare_mask=fmask, quality=lc.get("quality"))
        amp_q.update({"segment": seg, "flux_column": fcol, "n_detected_flares": len(det),
                      "finder_sigma": dinfo.get("sigma")})
        amp_by_seg[str(seg)] = amp_q
        matched = match_flares(seg_tests, det, cadence_days=cad)
        tpf = next((p for p in tpfs if seg is None or p.get("segment") is None
                    or int(p["segment"]) == int(seg)), None)
        # census on this segment's aperture
        sources_px: list[dict] = []
        if tpf is not None and srcs and tidx >= 0 and tpf.get("pix_fn") is not None:
            for s in srcs:
                try:
                    x, y = tpf["pix_fn"](s["ra"], s["dec"])
                except Exception:                         # noqa: BLE001
                    x, y = float("nan"), float("nan")
                sources_px.append({**s, "x": float(x), "y": float(y)})
        for m in matched:
            fr = {k: m.get(k) for k in ("energy_erg", "energy_raw", "t_start", "t_end", "t_peak",
                                        "segment", "above", "xi_conservative_stage1")}
            fr.update({"match_method": m.get("match_method"), "rank": m.get("rank"),
                       "flux_column": fcol, "outcome": OUTCOME_UNTESTABLE, "reason": ""})
            d = m.get("match")
            if d is None:
                fr["reason"] = "catalogue flare not re-detected in the light curve"
                flare_recs.append(fr)
                continue
            fr.update({"detected_t_start": d["t_start"], "detected_t_peak": d["t_peak"],
                       "detected_t_end": d["t_end"], "amplitude": d["amplitude"],
                       "peak_sigma": d["peak_sigma"], "ed_s": d["ed_s"],
                       "n_points": d["n_points"], "t_rise_days": d["t_rise_days"],
                       "t_decay_days": d["t_decay_days"]})
            es = flare_energy_shibayama(d["times"], d["rel_excess"], teff_k=teff, radius_rsun=rad,
                                        t_flare_k=float(params.t_flare_k), mission=mission,
                                        cadence_days=cad)
            fr["e_remeasured_bol_erg"] = es["e_bol_erg"]
            fr["e_remeasured_band_erg"] = flare_energy_band(d["ed_s"], teff_k=teff,
                                                            radius_rsun=rad, mission=mission)
            fr["e_ratio_remeasured_over_catalogue"] = (es["e_bol_erg"] / m["energy_erg"]
                                                       if m["energy_erg"] > 0 else float("nan"))
            remeasured.append(es["e_bol_erg"])
            if tpf is None:
                fr["reason"] = "no pixel file for this segment"
                flare_recs.append(fr)
                continue
            if not sources_px or tidx < 0:
                fr["reason"] = ("no Gaia census / no WCS: the flare's position cannot be "
                                "compared with any source")
            inm, bm, core = flare_cadence_masks(
                tpf["time"], d["t_start"], d["t_end"], cadence_days=cad,
                baseline_days=params.baseline_days, exclude_cadences=params.exclude_cadences,
                quality=tpf.get("quality"),
                core_times=[tt for tt, x in zip(d["times"], d["rel_excess"], strict=True)
                            if x >= float(params.core_fraction) * d["amplitude"]])
            sh = centroid_shift(tpf["time"], tpf["flux"], tpf["aperture"], inm, bm,
                                trend_order=params.trend_order)
            diff = difference_image(tpf["time"], tpf["flux"], core, bm, err_cube=tpf.get("flux_err"),
                                    trend_order=params.trend_order)
            fr.update({"n_in_flare_cadences": int(inm.sum()), "n_core_cadences": int(core.sum()),
                       "n_baseline_cadences": int(bm.sum()), "aperture_excess_a": sh.get("a"),
                       "shift_dx_px": sh.get("dx"), "shift_dy_px": sh.get("dy"),
                       "shift_dx_err_px": sh.get("dx_err"), "shift_dy_err_px": sh.get("dy_err"),
                       "baseline_x": sh.get("x0"), "baseline_y": sh.get("y0"),
                       "diff_snr_peak": diff.get("snr_peak"), "aperture_source": tpf.get("aperture_source")})
            if not sources_px or tidx < 0:
                flare_recs.append(fr)
                continue
            a_ap = _f(sh.get("a"))
            cen, csum = census(sources_px, target_index=tidx, aperture=tpf["aperture"],
                               prf_sigma_px=prf_sigma, aperture_amplitude=a_ap,
                               max_neighbour_amplitude=params.max_neighbour_amplitude)
            anc = anchor_sources(cen, target_index=tidx, baseline_xy=(sh.get("x0"), sh.get("y0")))
            att = attribute_flare(diff, anc, target_index=tidx, sys_floor_px=params.sys_floor_px,
                                  n_sigma=params.n_sigma_centroid, min_snr=params.min_snr,
                                  shift=sh, unresolved_px=params.unresolved_px)
            fr.update({"outcome": att["outcome"], "reason": att["reason"],
                       "diff_x": att["x_diff"], "diff_y": att["y_diff"],
                       "diff_sigma_px": att["sigma_px"], "z_target": att["z_target"],
                       "d_target_px": att["d_target_px"],
                       "d_target_arcsec": (att["d_target_px"] * tpf["pixel_scale_arcsec"]
                                           if np.isfinite(_f(att["d_target_px"])) else float("nan")),
                       "neighbour": att.get("neighbour"),
                       "consistent_neighbours": att["consistent_neighbours"],
                       "excluded_neighbours": att["excluded_neighbours"],
                       "unresolved_neighbours": att.get("unresolved_neighbours", []),
                       "wcs_anchor_dx_px": anc[tidx].get("anchor_dx"),
                       "wcs_anchor_dy_px": anc[tidx].get("anchor_dy"),
                       "sources": att["sources"], "census_summary": csum})
            for c in cen:
                census_rows.append({"star_key": key, "segment": seg, "flare_t_peak": d["t_peak"],
                                    **{k: c.get(k) for k in ("id", "gmag", "sep_arcsec", "pa_deg",
                                                             "is_target", "flux_ratio_to_target",
                                                             "capture_fraction", "flux_fraction",
                                                             "required_amplitude",
                                                             "excluded_by_arithmetic")}})
            flare_recs.append(fr)
        del order
    rec["flares"] = flare_recs
    rec["amplitude_by_segment"] = amp_by_seg
    rec["census_rows"] = census_rows

    # --- 6. the analytic census (one pixel and the 12" aperture) ---------------------
    if srcs and tidx >= 0:
        scale = TESS_PIXEL_ARCSEC if mission == "tess" else KEPLER_PIXEL_ARCSEC
        an = []
        for s in srcs:
            an.append({**s, "x": s["sep_arcsec"] / scale * math.cos(math.radians(s["pa_deg"])),
                       "y": s["sep_arcsec"] / scale * math.sin(math.radians(s["pa_deg"]))})
        a_ref = max([_f(f.get("aperture_excess_a")) for f in flare_recs
                     if np.isfinite(_f(f.get("aperture_excess_a")))] or [float("nan")])
        if not np.isfinite(a_ref):
            a_ref = max([_f(f.get("amplitude")) for f in flare_recs
                         if np.isfinite(_f(f.get("amplitude")))] or [float("nan")])
        for name, radius in (("one_pixel", params.pixel_radius_arcsec),
                             ("aperture", params.census_radius_arcsec)):
            c, s = census(an, target_index=tidx, aperture=None, prf_sigma_px=prf_sigma,
                          aperture_amplitude=a_ref, radius_px=float(radius) / scale,
                          max_neighbour_amplitude=params.max_neighbour_amplitude)
            rec["census"][name] = {"radius_arcsec": float(radius), "summary": s,
                                   "sources": [{k: x.get(k) for k in
                                                ("id", "gmag", "sep_arcsec", "pa_deg", "is_target",
                                                 "flux_ratio_to_target", "capture_fraction",
                                                 "flux_fraction", "required_amplitude",
                                                 "excluded_by_arithmetic")} for x in c]}

    # --- 7. xi on measured parameters ---------------------------------------------
    amp_q = {}
    if amp_by_seg:
        best = max(amp_by_seg.values(), key=lambda a: _f(a.get("rvar")) if np.isfinite(_f(a.get("rvar"))) else -1)
        amp_q = best
    rec["xi"] = _xi_block(entry, rows, prm, amp_q, remeasured, params)

    # --- 8. the verdict -------------------------------------------------------------
    v, why = star_verdict([f.get("outcome") for f in flare_recs])
    if v == VERDICT_UNTESTABLE:
        reasons = sorted({str(f.get("reason", "")) for f in flare_recs if f.get("reason")})
        if not tests:
            why = "no catalogue flare above the ceiling to test"
        elif reasons:
            why = "; ".join(reasons)[:300]
        if rec["statuses"].get("pixels") == STATUS_FAILED and "budget" in \
                str(log.stages[-1].get("error", "")) if log.stages else False:
            why = "budget_exhausted: " + why
    rec["verdict"], rec["verdict_reason"] = v, why
    rec["n_flares_tested"] = int(sum(1 for f in flare_recs if f.get("outcome") not in
                                     (OUTCOME_UNTESTABLE,)))
    rec["outcomes"] = {}
    for f in flare_recs:
        rec["outcomes"][f.get("outcome")] = rec["outcomes"].get(f.get("outcome"), 0) + 1
    rec["elapsed_s"] = round(own.elapsed(), 1)
    return rec


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------
def stage2_probe(conf: dict, out: Path, *, params: Stage2Params | None = None) -> dict:
    params = params or Stage2Params.from_config(conf)
    short = load_shortlist(Path(conf.get("_arc_dir", "results/arc")), params=params)
    rep = {"stage": "probe", "generated_utc": _now(), "mast": mast_probe(),
           "shortlist_size": len(short),
           "shortlist": [{k: s.get(k) for k in ("star_key", "tier", "catalogue",
                                                "xi_conservative_max", "n_above_conservative")}
                         for s in short],
           "params": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in params.__dict__.items()}}
    _write(out / "probe.json", rep)
    print(f"[arc-stage2] probe: MAST route = {rep['mast'].get('route_preferred')}; "
          f"{len(short)} stars shortlisted")
    return rep


def _overall_verdict(stars: list[dict]) -> tuple[str, str]:
    prime = [s for s in stars if s.get("tier") in ("interest", "candidate")] or stars
    if not stars:
        return VERDICT_S2_NO_DATA, "no star was shortlisted"
    tested = [s for s in prime if s.get("verdict") != VERDICT_UNTESTABLE]
    measured = [s for s in prime if (s.get("xi") or {}).get("params_measured")]
    if not tested and not measured:
        return VERDICT_S2_NO_DATA, ("no shortlisted star reached a pixel test or a measured "
                                    "parameter set")
    on_t = [s for s in prime if s.get("verdict") == VERDICT_ON_TARGET
            and (s.get("xi") or {}).get("xi_status") == "above_ceiling"]
    if on_t:
        return VERDICT_S2_ON_TARGET, (f"{len(on_t)} star(s) with every attributable flare on the "
                                      "target AND xi > 0 on measured parameters: "
                                      + ", ".join(s["star_key"] for s in on_t))
    on_n = [s for s in prime if s.get("verdict") == VERDICT_ON_NEIGHBOUR]
    if on_n and all(s.get("verdict") in (VERDICT_ON_NEIGHBOUR, VERDICT_UNTESTABLE)
                    or (s.get("xi") or {}).get("xi_status") != "above_ceiling" for s in prime):
        return VERDICT_S2_NEIGHBOUR, (f"{len(on_n)} star(s) whose exceeding flare sits on a Gaia "
                                      "neighbour: " + ", ".join(s["star_key"] for s in on_n))
    below = [s for s in prime if (s.get("xi") or {}).get("xi_status") == "below_ceiling"]
    if below and len(below) == len(prime):
        return VERDICT_S2_DISSOLVED, ("every interest star falls below the conservative ceiling "
                                      "once the measured radius / Teff and the star's own "
                                      "amplitude replace stage 1's assumptions")
    if not tested:
        return VERDICT_S2_UNTESTABLE, "no interest star reached an attributable flare"
    return VERDICT_S2_AMBIGUOUS, "the pixel test neither confirmed nor excluded the target"


def stage2_run(conf: dict, out: Path, *, params: Stage2Params | None = None, query_fn=None,
               cone_fn=None, lc_fn=None, tpf_fn=None, shortlist: list[dict] | None = None,
               log: AcquisitionLog | None = None) -> dict:
    params = params or Stage2Params.from_config(conf)
    log = log or AcquisitionLog(prefix="arc/stage2")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arc_dir = Path(conf.get("_arc_dir", "results/arc"))
    short = shortlist if shortlist is not None else load_shortlist(arc_dir, params=params)
    deadline = Deadline(budget_s=float(params.budget_s) if params.budget_s else None)
    stars: list[dict] = []
    flare_rows: list[dict] = []
    census_rows: list[dict] = []
    budget_hit = False
    for entry in short:
        if deadline.expired():
            budget_hit = True
            stars.append({"star_key": entry.get("star_key"), "star_id": entry.get("star_id"),
                          "mission": entry.get("mission"), "tier": entry.get("tier"),
                          "verdict": VERDICT_UNTESTABLE, "verdict_reason": "budget_exhausted",
                          "statuses": {}, "degraded": ["budget_exhausted"], "flares": [],
                          "xi": {"xi_status": "not_recomputed", "params_measured": False,
                                 "xi_conservative_stage1": _f(entry.get("xi_conservative_max"))}})
            continue
        try:
            rec = analyse_star(entry, conf=conf, params=params, query_fn=query_fn, cone_fn=cone_fn,
                               lc_fn=lc_fn, tpf_fn=tpf_fn, log=log, deadline=deadline)
        except Exception as exc:                          # noqa: BLE001
            log.record(f"analyse_{entry.get('star_key')}", "analyse_star", error=repr(exc)[:600])
            rec = {"star_key": entry.get("star_key"), "star_id": entry.get("star_id"),
                   "mission": entry.get("mission"), "tier": entry.get("tier"),
                   "verdict": VERDICT_UNTESTABLE, "verdict_reason": f"error: {exc!r}"[:300],
                   "statuses": {}, "degraded": ["analysis_error"], "flares": [],
                   "xi": {"xi_status": "not_recomputed", "params_measured": False,
                          "xi_conservative_stage1": _f(entry.get("xi_conservative_max"))}}
        for f in rec.get("flares") or []:
            flare_rows.append({"star_key": rec["star_key"], "tier": rec.get("tier"),
                               **{k: v for k, v in f.items() if k not in ("sources", "census_summary")}})
        census_rows.extend(rec.pop("census_rows", []) or [])
        stars.append(rec)
        # checkpoint after every star
        _write(out / "stars.json", {"generated_utc": _now(), "stars": stars})
        print(f"[arc-stage2] {rec['star_key']} ({rec.get('tier')}): {rec['verdict']} — "
              f"{rec.get('verdict_reason', '')[:120]}; xi_measured="
              f"{(rec.get('xi') or {}).get('xi_conservative_measured')}")
    verdict, reason = _overall_verdict(stars)
    counts = {}
    for s in stars:
        counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1
    xi_counts = {}
    for s in stars:
        k = (s.get("xi") or {}).get("xi_status", "not_recomputed")
        xi_counts[k] = xi_counts.get(k, 0) + 1
    summary = {
        "verdict": verdict, "reason": reason, "generated_utc": _now(),
        "n_shortlisted": len(short), "n_stars": len(stars),
        "n_interest": int(sum(1 for s in stars if s.get("tier") in ("interest", "candidate"))),
        "n_watch": int(sum(1 for s in stars if s.get("tier") == "watch")),
        "verdict_counts": counts, "xi_status_counts": xi_counts,
        "n_flares_examined": len(flare_rows),
        "flare_outcome_counts": _count([f.get("outcome") for f in flare_rows]),
        "n_params_measured": int(sum(1 for s in stars if (s.get("xi") or {}).get("params_measured"))),
        "budget_exhausted": bool(budget_hit or deadline.expired()),
        "elapsed_s": round(deadline.elapsed(), 1), "budget_s": params.budget_s,
        "stars": [{k: s.get(k) for k in ("star_key", "tier", "verdict", "verdict_reason",
                                         "n_flares_tested", "outcomes", "statuses", "degraded")}
                  | {"xi_conservative_stage1": (s.get("xi") or {}).get("xi_conservative_stage1"),
                     "xi_conservative_measured": (s.get("xi") or {}).get("xi_conservative_measured"),
                     "xi_status": (s.get("xi") or {}).get("xi_status"),
                     "teff_k": (s.get("params") or {}).get("teff_k"),
                     "radius_rsun": (s.get("params") or {}).get("radius_rsun"),
                     "radius_source": (s.get("params") or {}).get("radius_source"),
                     "amplitude_used": (s.get("xi") or {}).get("amplitude_used"),
                     "amplitude_used_source": (s.get("xi") or {}).get("amplitude_used_source")}
                  for s in stars],
        "degraded": sorted({d for s in stars for d in (s.get("degraded") or [])}),
        "acquisition": log.as_dict(),
        "mast": mast_probe(),
        "note": ("flare_on_target is a PIXEL-LEVEL statement about which Gaia-resolved source "
                 "brightened; a companion inside ~0.1\" (unresolved by Gaia and by the pixels "
                 "alike) is not excluded by it, and only spectroscopy or the flare colour "
                 "could; STAGE2_NO_DATA_REACHED is not a null result and is not written up "
                 "(CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    _write(out / "stars.json", {"generated_utc": summary["generated_utc"], "stars": stars})
    pd.DataFrame(flare_rows).to_csv(out / "flares.csv", index=False)
    pd.DataFrame(census_rows).to_csv(out / "census.csv", index=False)
    log.write(out / "acquisition_log.json")
    print(f"[arc-stage2] {verdict}: {reason}")
    return summary


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti arc-stage2",
                                description="ARC stage 2: is the flare on the target?")
    p.add_argument("--stage", default="all", choices=["probe", "all"])
    p.add_argument("--arc-dir", default="results/arc", help="stage 1 results directory")
    p.add_argument("--out-dir", default="results/arc/stage2")
    p.add_argument("--tiers", default="", help="comma-separated tiers (default from config)")
    p.add_argument("--max-stars", type=int, default=-1)
    a = p.parse_args(argv)
    conf = load_arc_config()
    conf["_arc_dir"] = a.arc_dir
    params = Stage2Params.from_config(conf)
    if a.tiers:
        params.tiers = tuple(x.strip() for x in a.tiers.split(",") if x.strip())
    if a.max_stars >= 0:
        params.max_stars = int(a.max_stars)
    out = Path(a.out_dir)
    stage2_probe(conf, out, params=params)
    if a.stage == "all":
        stage2_run(conf, out, params=params)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["Deadline", "Stage2Params", "VERDICT_S2_AMBIGUOUS", "VERDICT_S2_DISSOLVED",
           "VERDICT_S2_NEIGHBOUR", "VERDICT_S2_NO_DATA", "VERDICT_S2_ON_TARGET",
           "VERDICT_S2_UNTESTABLE", "analyse_star", "default_pixels_fn", "fetch_flare_rows",
           "fetch_gaia_flame", "gaia_sources", "lc_from_pixels", "lightkurve_lc_fn",
           "lightkurve_pixels_fn", "load_shortlist", "main", "mast_fits_pixels_fn", "mast_probe",
           "read_pixel_fits", "stage2_probe", "stage2_run", "stellar_parameters", "_clean_id"]

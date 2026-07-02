"""Aggregate the multi-cone clustering sweep into one quantified statement.

A single cone gives one p-value per space; a sweep of independent cones needs the
right combination.  Two questions, two statistics:

* **Is there a *global* excess of clustering** across the sampled volume?  ->
  Fisher's method combines the per-cone p-values into one chi-square.
* **Does *any single* cone over-cluster** (a localised population)?  -> the
  smallest per-cone p-value with a Bonferroni/Sidak trials correction for having
  looked in N cones.

Both are reported for position, velocity and phase space, so the sweep reads as a
single honest result (a quantified null, or a flagged cone) rather than a pile of
per-cone JSONs.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def _fisher(pvals: list[float]) -> tuple[float, float]:
    """Fisher's combined test: X2 = -2 sum ln p, dof = 2N.  Returns (X2, p)."""
    p = np.clip(np.asarray([x for x in pvals if np.isfinite(x)], float), 1e-12, 1.0)
    if not len(p):
        return float("nan"), float("nan")
    x2 = float(-2.0 * np.sum(np.log(p)))
    try:
        from scipy.stats import chi2
        return x2, float(chi2.sf(x2, 2 * len(p)))
    except Exception:  # noqa: BLE001
        return x2, float("nan")


def _sidak_min(pvals: list[float]) -> tuple[float, float]:
    """Smallest p with a Sidak trials correction: 1-(1-p_min)^N."""
    p = np.asarray([x for x in pvals if np.isfinite(x)], float)
    if not len(p):
        return float("nan"), float("nan")
    pmin = float(np.min(p))
    return pmin, float(1.0 - (1.0 - pmin) ** len(p))


def aggregate_sweep(root, alpha: float = 0.05) -> dict:
    """Read every committed cone summary and combine.  Returns the aggregate dict
    and writes ``results/cluster/AGGREGATE.json``."""
    root = Path(root)
    base = root / "results" / "cluster"
    cones = []
    for f in sorted(glob.glob(str(base / "f*" / "summary.json"))):
        d = json.loads(Path(f).read_text())
        cones.append({
            "field": d.get("field"),
            "n_searched": d.get("n_searched"),
            "n_ir_excess": d.get("n_ir_excess"),
            "p_pos": (d.get("clustering") or {}).get("p_value"),
            "p_vel": (d.get("clustering_velocity") or {}).get("p_value"),
            "p_phase": (d.get("clustering_phase_space") or {}).get("p_value"),
        })
    out = {"n_cones": len(cones),
           "total_stars": int(sum(c["n_searched"] or 0 for c in cones)),
           "total_ir_excess": int(sum(c["n_ir_excess"] or 0 for c in cones)),
           "cones": cones}
    for space in ("p_pos", "p_vel", "p_phase"):
        ps = [c[space] for c in cones if c.get(space) is not None]
        fx2, fp = _fisher(ps)
        pmin, psidak = _sidak_min(ps)
        out[space] = {
            "fisher_chi2": fx2, "fisher_p": fp,
            "min_p": pmin, "sidak_p": psidak,
            "global_over_clustered": bool(np.isfinite(fp) and fp < alpha),
            "any_cone_significant": bool(np.isfinite(psidak) and psidak < alpha),
        }
    # Headline: a detection needs a cone significant after the trials correction in
    # velocity or phase space (the discriminating spaces).
    out["detection"] = bool(out["p_vel"]["any_cone_significant"]
                            or out["p_phase"]["any_cone_significant"]
                            or out["p_vel"]["global_over_clustered"]
                            or out["p_phase"]["global_over_clustered"])
    base.mkdir(parents=True, exist_ok=True)
    (base / "AGGREGATE.json").write_text(json.dumps(out, indent=2, default=str))
    return out


__all__ = ["aggregate_sweep"]

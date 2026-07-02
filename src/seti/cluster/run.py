"""End-to-end population-clustering search over a Gaia x AllWISE volume.

The technosignature hypothesis under test: a civilisation that builds waste-heat
(infrared-excess) structures and expands to neighbouring systems would leave an
infrared-excess population that is **over-clustered in phase space** relative to
ordinary stars.  Every single-object excess is degenerate with dust/blends (this
project proved that six times over), but a *spatial/kinematic over-density* of the
excess tail, measured against a magnitude/colour/distance-matched random null, is
not reproducible by those single-object contaminants.

Pipeline: pull a Gaia x AllWISE volume, compute a W1-W2 infrared-excess indicator,
select the excess tail, and run the matched-null clustering test + a
friends-of-friends group finder over 3D position (and, with proper motions, phase
space).  Acquisition runs on the GitHub runner; the statistics are unit-tested
offline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .clustering import friends_of_friends, matched_null_clustering
from .phase_space import galactic_xyz, tangential_velocity

# Two-step, robust acquisition.  A single all-in-one 3-table join to the DR1
# AllWISE catalogue (string-keyed) makes the Gaia TAP server drop the result
# ("Error 500: cannot find result"), so instead: (1) a light single-table Gaia
# cone query for the stars, then (2) the WISE photometry in small source_id chunks
# via the official crossmatch.  Each query is cheap and reliable.
_STAR_QUERY = """
SELECT TOP {limit}
       g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error,
       g.pmra, g.pmdec, g.phot_g_mean_mag, g.bp_rp, g.ruwe
FROM gaiadr3.gaia_source AS g
WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec),
                 CIRCLE('ICRS', {ra}, {dec}, {radius}))
  AND g.parallax > {plx_min}
  AND g.parallax_over_error > 10
  AND g.phot_g_mean_mag < {g_max}
  AND g.ruwe < 1.4
"""

_WISE_QUERY = """
SELECT xm.source_id, w.w1mpro, w.w2mpro
FROM gaiadr3.allwise_best_neighbour AS xm
JOIN gaiadr1.allwise_original_valid AS w
  ON w.designation = xm.original_ext_source_id
WHERE xm.source_id IN ({ids})
"""


def _fetch_wise_for(source_ids, chunk: int = 4000) -> pd.DataFrame:
    from astroquery.gaia import Gaia
    frames = []
    ids = [int(s) for s in source_ids]
    for i in range(0, len(ids), chunk):
        sub = ",".join(str(s) for s in ids[i:i + chunk])
        try:
            r = (Gaia.launch_job_async(_WISE_QUERY.format(ids=sub))
                 .get_results().to_pandas())
            frames.append(r.rename(columns={c: c.lower() for c in r.columns}))
        except Exception as exc:  # noqa: BLE001
            print(f"[cluster] WISE chunk {i // chunk} failed: {exc!r}")
    if not frames:
        return pd.DataFrame(columns=["source_id", "w1mpro", "w2mpro"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("source_id")


def _fetch(ra: float, dec: float, radius_deg: float, plx_min: float, g_max: float,
           limit: int) -> pd.DataFrame:
    from astroquery.gaia import Gaia
    q = _STAR_QUERY.format(limit=int(limit), ra=ra, dec=dec, radius=radius_deg,
                           plx_min=plx_min, g_max=g_max)
    stars = Gaia.launch_job_async(q).get_results().to_pandas()
    stars = stars.rename(columns={c: c.lower() for c in stars.columns})
    print(f"[cluster] {len(stars)} Gaia stars in cone; fetching WISE...")
    wise = _fetch_wise_for(stars["source_id"].tolist())
    print(f"[cluster] {len(wise)} of {len(stars)} have an AllWISE match")
    return stars.merge(wise, on="source_id", how="inner")


def ir_excess_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """W1-W2 infrared-excess score.  Normal stellar photospheres have W1-W2 ~ 0
    (Rayleigh-Jeans); warm circumstellar/engineered emission makes W1-W2 > 0.  We
    take the excess *relative to the colour-dependent stellar locus* (the median
    W1-W2 at each BP-RP) so intrinsically red stars are not all flagged.
    """
    out = df.copy()
    w1 = pd.to_numeric(out.get("w1mpro"), errors="coerce")
    w2 = pd.to_numeric(out.get("w2mpro"), errors="coerce")
    bp_rp = pd.to_numeric(out.get("bp_rp"), errors="coerce")
    w1w2 = w1 - w2
    out["w1_w2"] = w1w2
    # Colour-binned median locus + robust scatter -> excess z-score.
    good = np.isfinite(w1w2) & np.isfinite(bp_rp)
    z = np.full(len(out), np.nan)
    if good.sum() > 50:
        try:
            bins = pd.qcut(bp_rp[good], q=min(20, good.sum() // 25),
                           labels=False, duplicates="drop")
        except Exception:  # noqa: BLE001
            bins = pd.cut(bp_rp[good], bins=10, labels=False)
        gidx = np.where(good)[0]
        tmp = pd.DataFrame({"i": gidx, "b": np.asarray(bins), "v": w1w2[good].to_numpy()})
        for _, grp in tmp.groupby("b"):
            med = np.median(grp["v"])
            mad = 1.4826 * np.median(np.abs(grp["v"] - med)) + 1e-6
            z[grp["i"].to_numpy()] = (grp["v"].to_numpy() - med) / mad
    out["ir_excess_z"] = z
    return out


def cluster_run(cfg: Config | None = None, ra: float = 200.0, dec: float = 0.0,
                radius_deg: float = 12.0, plx_min: float = 2.0, g_max: float = 16.0,
                limit: int = 200000, excess_z_min: float = 4.0,
                link_pc: float = 8.0, table: pd.DataFrame | None = None) -> dict:
    """Fetch a Gaia x AllWISE cone volume, flag the IR-excess tail, and test
    whether it is over-clustered in 3D position vs a matched random null.
    ``table`` may be supplied for offline tests instead of querying Gaia."""
    cfg = cfg or load_config()
    raw = table if table is not None else _fetch(ra, dec, radius_deg, plx_min,
                                                 g_max, limit)
    df = ir_excess_indicator(raw)
    df = galactic_xyz(df)
    df = tangential_velocity(df)

    mask = (pd.to_numeric(df["ir_excess_z"], errors="coerce") >= excess_z_min).to_numpy()
    mask = mask & np.isfinite(df[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)).all(1)
    n_excess = int(mask.sum())

    space = ["X_pc", "Y_pc", "Z_pc"]
    res = matched_null_clustering(df, mask, space, n_null=500)
    # Friends-of-friends in raw parsecs: link excess sources within link_pc.
    labels = (friends_of_friends(df[mask], space, linking_length=link_pc,
                                 min_size=4, standardize=False)
              if n_excess else pd.Series([], dtype=int))

    groups = []
    if len(labels):
        ex = df[mask].reset_index(drop=True)
        ex["group"] = labels.to_numpy()
        for gid, grp in ex[ex["group"] >= 0].groupby("group"):
            groups.append({
                "group": int(gid), "n": int(len(grp)),
                "ra": float(grp["ra"].mean()), "dec": float(grp["dec"].mean()),
                "dist_pc": float(grp["dist_pc"].median()),
                "vtan_kms": float(grp["vtan_kms"].median()),
                "source_ids": ([int(s) for s in grp["source_id"].head(30)]
                               if "source_id" in grp.columns else []),
            })
        groups.sort(key=lambda d: d["n"], reverse=True)

    out_dir = cfg.root / "results" / "cluster"
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_excess:
        cols = [c for c in ("source_id", "ra", "dec", "parallax", "dist_pc",
                            "pmra", "pmdec", "vtan_kms", "phot_g_mean_mag", "bp_rp",
                            "w1_w2", "ir_excess_z") if c in df.columns]
        df[mask][cols].to_csv(out_dir / "ir_excess_tail.csv", index=False)
    if groups:
        pd.DataFrame(groups).to_csv(out_dir / "clustered_groups.csv", index=False)

    summary = {
        "field": {"ra": ra, "dec": dec, "radius_deg": radius_deg},
        "n_searched": int(len(df)), "n_ir_excess": n_excess,
        "excess_z_min": excess_z_min,
        "clustering": {k: res.get(k) for k in
                       ("S_obs", "S_null_mean", "S_null_std", "z", "p_value",
                        "over_clustered", "n_anom")},
        "n_groups": len(groups),
        "top_groups": groups[:15],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[cluster]", json.dumps({"n_searched": summary["n_searched"],
          "n_ir_excess": n_excess, "p_value": res.get("p_value"),
          "over_clustered": res.get("over_clustered"), "n_groups": len(groups)}))
    return summary


__all__ = ["cluster_run", "ir_excess_indicator"]

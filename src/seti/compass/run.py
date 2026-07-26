"""COMPASS runner: pole field -> coherence scan -> banded-shuffle null -> report."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .axial import (
    bingham_stats_batch,
    ecliptic_latitude_deg,
    group_matrix,
    pole_axes,
    principal_axis,
    shuffle_axes_within_bands,
)
from .orbit import thiele_innes_to_geometric

SEED = 20260726


def build_field(nss: pd.DataFrame) -> pd.DataFrame:
    """Positions (Galactic pc), pole axes, and discriminator columns."""
    a0, inc, node, _ = thiele_innes_to_geometric(
        nss["a_thiele_innes"], nss["b_thiele_innes"],
        nss["f_thiele_innes"], nss["g_thiele_innes"])
    ok = np.isfinite(inc) & np.isfinite(node) & (a0 > 0)
    df = nss.loc[ok].reset_index(drop=True).copy()
    df["a0_mas"] = np.asarray(a0)[ok]
    df["inclination_deg"] = np.asarray(inc)[ok]
    df["node_deg"] = np.asarray(node)[ok]

    axes = pole_axes(df["ra"], df["dec"],
                     df["inclination_deg"], df["node_deg"])
    df[["pole_x", "pole_y", "pole_z"]] = axes

    from .axial import _ICRS_TO_GAL, tangent_basis
    _, _, los = tangent_basis(df["ra"], df["dec"])
    dist_pc = 1000.0 / df["parallax"].to_numpy(float)
    pos = (los @ _ICRS_TO_GAL.T) * dist_pc[:, None]
    df[["x_pc", "y_pc", "z_pc"]] = pos
    df["ecl_lat_deg"] = ecliptic_latitude_deg(df["ra"], df["dec"])
    return df


def _vet(df: pd.DataFrame, members: list[int]) -> dict:
    """Co-natal / co-moving discriminators for one patch."""
    g = df.iloc[members]
    mh = pd.to_numeric(g["mh_gspphot"], errors="coerce").dropna()
    mh_mad = float((mh - mh.median()).abs().median()) if len(mh) >= 4 else None
    # Tangential velocity spread (km/s); 4.74 km/s per mas/yr at 1 kpc.
    d_kpc = 1.0 / g["parallax"].to_numpy(float)
    vt_ra = 4.74 * g["pmra"].to_numpy(float) * d_kpc
    vt_de = 4.74 * g["pmdec"].to_numpy(float) * d_kpc
    v = np.stack([vt_ra, vt_de], 1)
    v_spread = float(np.sqrt(((v - np.nanmedian(v, 0)) ** 2)
                             .sum(1)[np.isfinite(v).all(1)].mean())) \
        if np.isfinite(v).all(1).any() else None
    return {
        "n_with_mh": int(len(mh)), "mh_mad_dex": mh_mad,
        "chem_conatal_possible": (mh_mad is not None and mh_mad < 0.05),
        "vt_spread_kms": v_spread,
        "comoving_possible": (v_spread is not None and v_spread < 10.0),
        "solution_types": sorted(g["nss_solution_type"].unique().tolist()),
    }


def compass_run(cfg: Config | None = None, stage: str = "all",
                radii_pc: tuple = (25.0, 50.0, 100.0), n_min: int = 8,
                n_shuffles: int = 200, band_deg: float = 5.0,
                sig_min: float = 10.0, poe_min: float = 5.0,
                d_max_pc: float = 2000.0, top_k: int = 10) -> dict:
    from .acquire import fetch_nss

    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "compass"
    out_dir.mkdir(parents=True, exist_ok=True)

    nss = fetch_nss(out_dir, sig_min=sig_min, poe_min=poe_min)
    if stage == "fetch":
        return {"stage": stage, "n_orbits": int(len(nss))}

    df = build_field(nss)
    df = df[np.linalg.norm(df[["x_pc", "y_pc", "z_pc"]], axis=1)
            <= d_max_pc].reset_index(drop=True)
    pos = df[["x_pc", "y_pc", "z_pc"]].to_numpy(float)
    axes = df[["pole_x", "pole_y", "pole_z"]].to_numpy(float)
    band = np.floor((df["ecl_lat_deg"].to_numpy(float) + 90.0)
                    / band_deg).astype(int)

    results, candidates = {}, []
    for radius in radii_pc:
        m, counts, centers = group_matrix(pos, radius_pc=float(radius),
                                          n_min=n_min)
        if m is None:
            results[f"r{radius:g}"] = {"n_groups": 0}
            continue
        real = bingham_stats_batch(m, counts, axes)
        order = np.argsort(-real)

        # Checkpointed shuffle null (positions/groups fixed; axes permuted
        # within ecliptic-latitude bands to preserve the scanning-law imprint).
        null_path = out_dir / f"null_r{radius:g}.json"
        null_max = json.loads(null_path.read_text()) \
            if null_path.exists() else []
        while len(null_max) < n_shuffles:
            sh = shuffle_axes_within_bands(
                axes, band, np.random.default_rng(SEED + 1000 + len(null_max)
                                                  + int(radius) * 100000))
            null_max.append(float(bingham_stats_batch(m, counts, sh).max()))
            if len(null_max) % 25 == 0:
                null_path.write_text(json.dumps(null_max))
                print(f"[compass] r={radius:g}: {len(null_max)}/{n_shuffles} "
                      "shuffles")
        null_path.write_text(json.dumps(null_max))

        real_max = float(real.max())
        p_global = (1 + sum(1 for s in null_max if s >= real_max)) \
            / (len(null_max) + 1)
        results[f"r{radius:g}"] = {
            "n_groups": int(len(counts)), "max_stat": real_max,
            "null_max_median": float(np.median(null_max)),
            "null_max_p99": float(np.percentile(null_max, 99)),
            "p_global": p_global,
        }
        print(f"[compass] r={radius:g}: groups {len(counts)}, real max "
              f"{real_max:.1f}, null med {np.median(null_max):.1f}, "
              f"p={p_global:.4f}")

        for gi in order[:top_k]:
            members = m.getrow(int(gi)).indices.tolist()
            rec = {
                "radius_pc": float(radius), "n": int(counts[gi]),
                "stat": float(real[gi]),
                "above_null_p99": bool(real[gi]
                                       > np.percentile(null_max, 99)),
                "axis_gal": [float(x)
                             for x in principal_axis(axes[members])],
                "center_source_id": int(df["source_id"].iloc[centers[gi]]),
                "member_source_ids": [int(s) for s in
                                      df["source_id"].iloc[members]],
                **_vet(df, members),
            }
            candidates.append(rec)

    summary = {
        "n_orbits_fetched": int(len(nss)), "n_field": int(len(df)),
        "d_max_pc": d_max_pc, "n_min": n_min, "n_shuffles": n_shuffles,
        "band_deg": band_deg, "sig_min": sig_min, "poe_min": poe_min,
        "radii": results,
        "verdict_rule": "a candidate requires p_global < 0.01 at its radius "
                        "AND above_null_p99 AND chem heterogeneous AND not "
                        "co-moving (docs/compass.md section 5)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "candidates.json").write_text(
        json.dumps({"top_per_radius": candidates}, indent=2))

    lines = ["# COMPASS run report", "",
             f"Field: {len(df)} astrometric orbits within {d_max_pc:.0f} pc "
             f"(of {len(nss)} fetched); {n_shuffles} scanning-law-banded "
             f"shuffles per radius.", "",
             "| radius (pc) | groups | real max S | null med | null p99 | p |",
             "|---|---|---|---|---|---|"]
    for r in radii_pc:
        v = results.get(f"r{r:g}", {})
        if v.get("n_groups"):
            lines.append(f"| {r:g} | {v['n_groups']} | {v['max_stat']:.1f} | "
                         f"{v['null_max_median']:.1f} | "
                         f"{v['null_max_p99']:.1f} | {v['p_global']:.4f} |")
    lines += ["", "Top patches per radius (with co-natal/co-moving "
              "discriminators) in candidates.json.", ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines))
    print(f"[compass] summary: {json.dumps(summary)[:300]}")
    return summary

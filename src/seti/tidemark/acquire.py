"""Wide-area parent-sample acquisition (runner only).

TIDEMARK consumes other channels' anomaly catalogues, but it must not be *blocked*
on them, and none of the existing channels publishes a parent sample wide enough
to carry a Galactic gradient anyway --- the ``cluster`` pilot covered three
12-degree cones.  A front test needs lever arm: several kpc in Galactocentric
radius, the full range of |z| the disk offers, and all Galactic longitudes.

So this module builds its own: a grid of cones spread over the sky, Gaia DR3 x
AllWISE, with **every** star retained (not just the anomalous tail) plus the
covariates that govern detectability.

Two design points that the test's validity depends on
-----------------------------------------------------
1. **Uniform, position-independent subsampling.**  Cones toward the inner Galaxy
   contain orders of magnitude more stars than cones toward the pole.  Capping
   them with ADQL ``TOP`` would impose an arbitrary, *undocumented* selection ---
   ``TOP`` without ``ORDER BY`` returns whatever the server finds first.  Instead
   we subsample on ``random_index``, Gaia's built-in uniform random permutation,
   which is exactly position-independent, exactly reproducible, and whose stride
   is recorded per star so the null can match on it.  Any cone that still hits
   the row cap is flagged ``truncated`` and excluded, because a truncated cone
   *is* an uncontrolled selection.
2. **The excess locus is fitted globally.**  ``seti.cluster.ir_excess_indicator``
   fits the stellar W1-W2 locus within whatever frame it is handed.  Run per
   cone, it normalises every field to its own median --- which would silently
   delete the field-to-field rate differences that are the entire signal.
   ``excess_axis`` therefore fits one locus across all cones at once.  This is
   the single most important line in the module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STAR_QUERY = """
SELECT TOP {cap}
       g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error,
       g.pmra, g.pmdec, g.radial_velocity, g.phot_g_mean_mag, g.bp_rp, g.ruwe,
       g.astrometric_n_good_obs_al, g.ebpminrp_gspphot, g.teff_gspphot,
       g.mh_gspphot, g.random_index
FROM gaiadr3.gaia_source AS g
WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec),
                 CIRCLE('ICRS', {ra}, {dec}, {radius}))
  AND g.parallax > {plx_min}
  AND g.parallax_over_error > 10
  AND g.phot_g_mean_mag < {g_max}
  AND g.ruwe < 1.4
  AND MOD(g.random_index, {stride}) = 0
"""

_WISE_QUERY = """
SELECT xm.source_id, w.w1mpro, w.w1sigmpro, w.w2mpro, w.w2sigmpro
FROM gaiadr3.allwise_best_neighbour AS xm
JOIN gaiadr1.allwise_original_valid AS w
  ON w.designation = xm.original_ext_source_id
WHERE xm.source_id IN ({ids})
"""

_GAL_TO_ICRS = np.array([
    [-0.0548755604162154, +0.4941094278755837, -0.8676661490190047],
    [-0.8734370902348850, -0.4448296299600112, -0.1980763734312015],
    [-0.4838350155487132, +0.7469822444972189, +0.4559837761750669],
])


def galactic_to_icrs(l_deg, b_deg) -> tuple[np.ndarray, np.ndarray]:
    l = np.radians(np.asarray(l_deg, float))
    b = np.radians(np.asarray(b_deg, float))
    u = np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=0)
    v = _GAL_TO_ICRS @ u
    ra = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    return ra, dec


def cone_grid(grid: str = "sparse") -> pd.DataFrame:
    """Cone centres, laid out in *Galactic* coordinates for maximum lever arm.

    The plane ring at |b| = 10 gives the Galactocentric-radius baseline (l = 0
    and l = 180 differ by ~2x the cone distance in R); the mid-latitude and polar
    rings give the |z| baseline; full longitude coverage gives the dipole.
    """
    rings = {
        "sparse": [(10.0, 12), (-10.0, 12), (35.0, 6), (-35.0, 6), (70.0, 3), (-70.0, 3)],
        "dense": [(8.0, 24), (-8.0, 24), (25.0, 12), (-25.0, 12), (45.0, 8),
                  (-45.0, 8), (70.0, 4), (-70.0, 4)],
        "plane": [(8.0, 18), (-8.0, 18)],
        "pilot": [(15.0, 6), (-15.0, 6)],
    }[grid]
    rows = []
    for b, n in rings:
        for k in range(n):
            l = 360.0 * k / n
            ra, dec = galactic_to_icrs(l, b)
            rows.append({"cone": len(rows), "l_centre": l, "b_centre": b,
                         "ra_centre": float(ra), "dec_centre": float(dec)})
    return pd.DataFrame(rows)


def _run_query(query: str, retries: int = 4) -> pd.DataFrame:
    """Reuses the ``cluster`` channel's hardened Gaia TAP wrapper (async with
    exponential backoff, synchronous fallback on the last attempt)."""
    from ..cluster.run import _run_query as _q
    return _q(query, retries=retries)


def _fetch_wise(source_ids, chunk: int = 2000) -> pd.DataFrame:
    frames = []
    ids = [int(s) for s in source_ids]
    for i in range(0, len(ids), chunk):
        sub = ",".join(str(s) for s in ids[i:i + chunk])
        try:
            frames.append(_run_query(_WISE_QUERY.format(ids=sub)))
        except Exception as exc:                                # noqa: BLE001
            print(f"[tidemark] WISE chunk {i // chunk} failed: {exc!r}")
    if not frames:
        return pd.DataFrame(columns=["source_id", "w1mpro", "w1sigmpro",
                                     "w2mpro", "w2sigmpro"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("source_id")


def fetch_cone(ra: float, dec: float, *, radius_deg: float = 6.0,
               plx_min: float = 1.0, g_max: float = 17.0, stride: int = 20,
               cap: int = 400000) -> pd.DataFrame:
    """One cone: Gaia rows (uniformly subsampled) joined to AllWISE photometry."""
    q = _STAR_QUERY.format(cap=int(cap), ra=ra, dec=dec, radius=radius_deg,
                           plx_min=plx_min, g_max=g_max, stride=int(stride))
    stars = _run_query(q)
    truncated = len(stars) >= cap
    if truncated:
        print(f"[tidemark] cone ({ra:.1f},{dec:.1f}) hit the row cap "
              f"({cap}); marking truncated and dropping it")
        stars = stars.iloc[:0]
    if not len(stars):
        return stars
    wise = _fetch_wise(stars["source_id"].tolist())
    out = stars.merge(wise, on="source_id", how="inner")
    out["sampling_stride"] = int(stride)
    out["truncated"] = bool(truncated)
    return out


def parent_sample(*, grid: str = "sparse", radius_deg: float = 6.0,
                  plx_min: float = 1.0, g_max: float = 17.0, stride: int = 20,
                  cap: int = 400000, limit: int | None = None,
                  cones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fetch the whole grid.  Returns *every* star searched, not a candidate list."""
    cones = cone_grid(grid) if cones is None else cones
    frames = []
    for _, c in cones.iterrows():
        try:
            df = fetch_cone(float(c["ra_centre"]), float(c["dec_centre"]),
                            radius_deg=radius_deg, plx_min=plx_min, g_max=g_max,
                            stride=stride, cap=cap)
        except Exception as exc:                                # noqa: BLE001
            print(f"[tidemark] cone {int(c['cone'])} failed: {exc!r}")
            continue
        if not len(df):
            continue
        df["cone"] = int(c["cone"])
        df["cone_l"] = float(c["l_centre"])
        df["cone_b"] = float(c["b_centre"])
        frames.append(df)
        print(f"[tidemark] cone {int(c['cone'])} (l={c['l_centre']:.0f}, "
              f"b={c['b_centre']:+.0f}): {len(df)} stars with AllWISE")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    if limit and len(out) > limit:
        out = out.sample(n=int(limit), random_state=1).reset_index(drop=True)
    return out


def excess_axis(table: pd.DataFrame, *, n_colour_bins: int = 25) -> pd.DataFrame:
    """Attach the IR-excess anomaly axis and the detectability covariates.

    **The locus is fitted once, over the whole table.**  Fitting it per cone
    would normalise each field to its own median and delete the very
    field-to-field rate differences TIDEMARK exists to measure.
    """
    out = table.copy()
    from .ingest import numeric
    w1 = numeric(out, "w1mpro")
    w2 = numeric(out, "w2mpro")
    col = numeric(out, "bp_rp")
    w1w2 = w1 - w2
    out["w1_w2"] = w1w2
    good = np.isfinite(w1w2) & np.isfinite(col)
    z = np.full(len(out), np.nan)
    if good.sum() > 200:
        edges = np.unique(np.quantile(col[good],
                                      np.linspace(0, 1, int(n_colour_bins) + 1)))
        idx = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, edges.size - 2)
        for bi in range(edges.size - 1):
            sel = good & (idx == bi)
            if sel.sum() < 25:
                continue
            v = w1w2[sel]
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
            z[sel] = (v - med) / mad
    out["ir_excess_z"] = z

    # Detectability covariates.
    s1 = numeric(out, "w1sigmpro")
    s2 = numeric(out, "w2sigmpro")
    out["w1w2_sigma"] = np.hypot(s1, s2)
    ebpminrp = numeric(out, "ebpminrp_gspphot")
    out["ebv"] = ebpminrp / 1.34            # E(BP-RP) ~ 1.34 E(B-V) for Gaia bands
    if "astrometric_n_good_obs_al" in out.columns:
        out["n_obs"] = pd.to_numeric(out["astrometric_n_good_obs_al"], errors="coerce")
    for src, dst in (("mh_gspphot", "feh"), ("teff_gspphot", "teff")):
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    from .ingest import add_galactic_frame, local_density
    out = add_galactic_frame(out)
    out["log_local_density"] = local_density(out)
    return out


__all__ = ["cone_grid", "galactic_to_icrs", "fetch_cone", "parent_sample",
           "excess_axis"]

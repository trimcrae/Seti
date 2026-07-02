"""End-to-end close-encounter search anchored on the hycean biosignature host.

Hypothesis under test.  K2-18 b is the hycean world with a JWST hint of a
biosignature (DMS/DMSO; Madhusudhan et al. 2023, 2025 -- contested, treated here
as the premise, not a result).  *If* something living arose there, the material
that could carry it to another star -- impact ejecta, dormant spores, free-flying
bodies of the 'Oumuamua/Borisov class -- is delivered most efficiently during a
**close, slow stellar encounter**, when two systems' outer reservoirs overlap and
the relative speed is low enough for capture.  Because the stellar neighbourhood
*reshuffles over time* (today's neighbours are not those of a few Myr ago), the
right question is not "who is nearby now" but "who passed close to K2-18, slowly,
in the recent past".

Pipeline (acquisition on the GitHub runner; maths unit-tested offline):

1. resolve K2-18's full 6D phase space from Gaia DR3 (RV essential);
2. pull every Gaia DR3 source with a radial velocity inside a heliocentric
   distance shell that brackets the search volume around K2-18;
3. build 6D phase space, compute each star's linear closest approach to K2-18,
   and score past close/slow encounters for transfer plausibility;
4. write the ranked recipient candidates + co-moving companions.

A surviving candidate is a *specific nearby star* that passed close to K2-18 at
low relative velocity within the viability window -- the star most likely, on
kinematic grounds alone, to have received K2-18-origin material.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .encounters import closest_approach, flag_comoving, transfer_score
from .kinematics import phase_space_6d

# K2-18 (EPIC 201912552 = 2MASS J11301441+0735371). The runner resolves the live
# Gaia DR3 6D vector by source_id; these are an approximate literature fallback
# only, used if acquisition fails, and are clearly superseded by the fetched row.
K2_18_SOURCE_ID = 3892950081412683520
K2_18_FALLBACK = {
    "source_id": K2_18_SOURCE_ID, "ra": 172.560055, "dec": 7.588391,
    "parallax": 26.30, "pmra": -84.68, "pmdec": -60.93,
    "radial_velocity": 0.35,   # km/s, systemic (literature); superseded by Gaia
}

_ANCHOR_QUERY = """
SELECT source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec,
       radial_velocity, phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
"""

# All-sky 6D shell: every star with a radial velocity whose heliocentric distance
# brackets the anchor's, so the search volume around the anchor is fully covered.
# The 3D cut to the anchor is applied in Python after the fetch.
_SHELL_QUERY = """
SELECT TOP {limit}
       source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec,
       radial_velocity, phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE parallax > {plx_min} AND parallax < {plx_max}
  AND parallax_over_error > 8
  AND radial_velocity IS NOT NULL
  AND ruwe < 1.4
"""


def _run_query(query: str, retries: int = 4) -> pd.DataFrame:
    """Robust Gaia ADQL (async with exp. backoff, synchronous fallback on the
    last try) -- the same pattern the clustering channel uses to survive the Gaia
    TAP server's intermittent async-result drops."""
    import time

    from astroquery.gaia import Gaia
    last = None
    for attempt in range(retries):
        try:
            if attempt == retries - 1:
                job = Gaia.launch_job(query)
            else:
                job = Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[panspermia] query attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gaia query failed after {retries} attempts: {last!r}")


def _resolve_anchor(source_id: int) -> dict:
    """Fetch K2-18's live 6D vector from Gaia DR3; fall back to the committed
    literature value only if the fetch fails or returns no radial velocity (the
    encounter maths is undefined without a 3D velocity)."""
    try:
        row = _run_query(_ANCHOR_QUERY.format(source_id=source_id))
        if len(row) and np.isfinite(pd.to_numeric(row["radial_velocity"],
                                                   errors="coerce").iloc[0]):
            r = row.iloc[0]
            print(f"[panspermia] resolved anchor from Gaia DR3: "
                  f"plx={r['parallax']:.3f} rv={r['radial_velocity']:.3f}")
            return {k: r[k] for k in ("source_id", "ra", "dec", "parallax",
                                      "pmra", "pmdec", "radial_velocity")}
        print("[panspermia] Gaia row lacks a radial velocity; using fallback")
    except Exception as exc:  # noqa: BLE001
        print(f"[panspermia] anchor resolve failed ({exc!r}); using fallback")
    return dict(K2_18_FALLBACK)


def _anchor_phase_space(anchor: dict) -> dict:
    ps = phase_space_6d(pd.DataFrame([anchor])).iloc[0]
    a = dict(anchor)
    a.update({k: float(ps[k]) for k in ("X_pc", "Y_pc", "Z_pc",
                                        "U_kms", "V_kms", "W_kms", "dist_pc")})
    return a


def _fetch_shell(anchor: dict, search_pc: float, g_max: float,
                 limit: int) -> pd.DataFrame:
    """Pull the all-sky 6D shell whose heliocentric distances bracket the
    ``search_pc`` sphere around the anchor."""
    d0 = anchor["dist_pc"]
    d_lo = max(d0 - search_pc, 1.0)
    d_hi = d0 + search_pc
    plx_min = 1000.0 / d_hi
    plx_max = 1000.0 / d_lo
    q = _SHELL_QUERY.format(limit=int(limit), plx_min=plx_min, plx_max=plx_max)
    df = _run_query(q)
    if g_max is not None and "phot_g_mean_mag" in df.columns:
        df = df[pd.to_numeric(df["phot_g_mean_mag"], errors="coerce") < g_max]
    print(f"[panspermia] {len(df)} Gaia 6D stars in the distance shell "
          f"[{d_lo:.1f}, {d_hi:.1f}] pc")
    return df.reset_index(drop=True)


def panspermia_run(cfg: Config | None = None, source_id: int = K2_18_SOURCE_ID,
                   search_pc: float = 40.0, g_max: float = 16.0,
                   limit: int = 400000, t_max_myr: float = 10.0,
                   d_min_max_pc: float = 2.0, anchor: dict | None = None,
                   table: pd.DataFrame | None = None) -> dict:
    """Rank the stars most likely to have received K2-18-origin material.

    ``anchor``/``table`` may be supplied for offline tests instead of querying
    Gaia.  ``search_pc`` is the 3D radius around K2-18 to consider; ``t_max_myr``
    the past-encounter viability window; ``d_min_max_pc`` the closest-approach
    cut that defines the shortlist.
    """
    cfg = cfg or load_config()

    anchor = anchor or _resolve_anchor(source_id)
    anchor = _anchor_phase_space(anchor)

    raw = table if table is not None else _fetch_shell(anchor, search_pc, g_max, limit)
    df = phase_space_6d(raw)
    # Keep only stars with a defined space velocity and inside the 3D search
    # sphere around K2-18 (and drop the anchor itself if present).
    df = closest_approach(anchor, df)
    good = (np.isfinite(df["v_rel_kms"].to_numpy())
            & (df["sep_now_pc"].to_numpy() <= search_pc)
            & (df.get("source_id", pd.Series(-1, index=df.index)).to_numpy()
               != anchor.get("source_id", -1)))
    df = df[good].reset_index(drop=True)

    df = transfer_score(df, t_max_myr=t_max_myr)
    df = flag_comoving(df)

    n_past = int(df["past_encounter"].sum())
    shortlist = df[(df["past_encounter"]) & (df["d_min_pc"] <= d_min_max_pc)].copy()
    shortlist = shortlist.sort_values("transfer_score", ascending=False)
    comoving = df[df["comoving"]].sort_values("v_rel_kms")

    out_dir = cfg.root / "results" / "panspermia"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("source_id", "ra", "dec", "dist_pc", "phot_g_mean_mag",
                        "bp_rp", "radial_velocity", "sep_now_pc", "v_rel_kms",
                        "t_enc_myr", "d_min_pc", "transfer_score", "comoving")
            if c in df.columns]
    if len(shortlist):
        shortlist[cols].to_csv(out_dir / "recipient_candidates.csv", index=False)
    if len(comoving):
        comoving[cols].to_csv(out_dir / "comoving_companions.csv", index=False)
    # Always write the full scored table (small) so a re-vet needs no re-query.
    df.sort_values("transfer_score", ascending=False)[cols].to_csv(
        out_dir / "encounters_all.csv", index=False)

    def _rec(r) -> dict:
        return {k: (float(r[k]) if k not in ("source_id",) else int(r[k]))
                for k in cols if k in r and pd.notna(r[k]) and k != "comoving"}

    summary = {
        "anchor": {"name": "K2-18", "source_id": int(anchor.get("source_id", 0)),
                   "dist_pc": round(float(anchor["dist_pc"]), 3),
                   "U_kms": round(float(anchor["U_kms"]), 3),
                   "V_kms": round(float(anchor["V_kms"]), 3),
                   "W_kms": round(float(anchor["W_kms"]), 3)},
        "search_pc": search_pc, "t_max_myr": t_max_myr, "d_min_max_pc": d_min_max_pc,
        "n_searched": int(len(df)),
        "n_past_encounter": n_past,
        "n_shortlist": int(len(shortlist)),
        "n_comoving": int(len(comoving)),
        "top_recipients": [_rec(r) for _, r in shortlist.head(20).iterrows()],
        "closest_approach_pc": (round(float(shortlist["d_min_pc"].min()), 4)
                                if len(shortlist) else None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[panspermia]", json.dumps({
        "n_searched": summary["n_searched"], "n_past": n_past,
        "n_shortlist": summary["n_shortlist"], "n_comoving": summary["n_comoving"],
        "d_min_min_pc": summary["closest_approach_pc"]}))
    return summary


__all__ = ["panspermia_run", "K2_18_SOURCE_ID", "K2_18_FALLBACK"]

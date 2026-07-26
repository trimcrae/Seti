"""Gaia DR3 6D acquisition for the HERDSMAN convergence search.

Pulls the RV-complete solar-neighbourhood sample in parallax shells (single
monolithic queries at these row counts time out on the TAP server), applies the
DR3 radial-velocity zero-point correction, and attaches each star's scalar
velocity uncertainty — the quantity that sets the detector's meeting radius
R(t) and therefore the search's usable horizon.

Selection notes, tied to the contamination model in ``docs/herdsman.md``:

* ``rv_template_teff < 8500`` — the published zero-point correction (Katz et
  al. 2023) applies to cool-template sources; hot-template DR3 RVs carry a
  separate, larger bias and are excluded rather than half-corrected.
* ``sigv`` includes an astrophysical floor (default 0.3 km/s) for the
  star-to-star scatter of gravitational redshift + convective blueshift, which
  no pipeline correction removes; without it the meeting radius would be
  optimistically small and true rendezvous would be missed.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

# Tangential-velocity constant: v_tan[km/s] = _K * mu[mas/yr] * d[kpc].
_K = 4.740470446

_QUERY = """
SELECT TOP {limit}
       source_id, ra, dec, parallax, parallax_error, parallax_over_error,
       pmra, pmra_error, pmdec, pmdec_error,
       radial_velocity, radial_velocity_error, rv_nb_transits, rv_template_teff,
       grvs_mag, phot_g_mean_mag, bp_rp, ruwe,
       mh_gspphot, teff_gspphot, logg_gspphot
FROM gaiadr3.gaia_source
WHERE parallax >= {plx_min} AND parallax < {plx_max}
  AND parallax_over_error > 10
  AND ruwe < 1.4
  AND radial_velocity IS NOT NULL
  AND radial_velocity_error < {rv_err_max}
  AND rv_template_teff < 8500
"""

# Distance-shell edges (pc) used to chunk the pull; trimmed to d_max at runtime.
_SHELL_EDGES_PC = [2000.0, 1600.0, 1300.0, 1050.0, 850.0, 700.0, 560.0, 480.0,
                   400.0, 320.0, 260.0, 210.0, 160.0, 110.0, 60.0, 0.0]


def _run_query(query: str, retries: int = 4, tag: str = "herdsman") -> pd.DataFrame:
    """Robust Gaia ADQL: async with exponential backoff, sync on the last try."""
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
            print(f"[{tag}] query attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gaia query failed after {retries} attempts: {last!r}")


def fetch_sample(d_max_pc: float = 300.0, rv_err_max_kms: float = 1.5,
                 g_max: float | None = 14.5,
                 limit_per_shell: int = 3_000_000) -> pd.DataFrame:
    """Pull the 6D quality sample within ``d_max_pc``, shell by shell."""
    edges = [e for e in _SHELL_EDGES_PC if e < d_max_pc] + [0.0]
    edges = sorted(set([d_max_pc] + edges), reverse=True)
    frames = []
    for d_hi, d_lo in zip(edges[:-1], edges[1:], strict=False):
        plx_min = 1000.0 / d_hi
        plx_max = 1000.0 / d_lo if d_lo > 0 else 1e9
        q = _QUERY.format(limit=int(limit_per_shell), plx_min=plx_min,
                          plx_max=min(plx_max, 1e4), rv_err_max=rv_err_max_kms)
        df = _run_query(q)
        if len(df) >= limit_per_shell:
            print(f"[herdsman] WARNING: shell [{d_lo:.0f},{d_hi:.0f}] pc hit the "
                  f"row limit ({limit_per_shell}); sample may be incomplete")
        print(f"[herdsman] shell [{d_lo:.0f},{d_hi:.0f}] pc: {len(df)} stars")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    if g_max is not None and "phot_g_mean_mag" in out.columns:
        out = out[pd.to_numeric(out["phot_g_mean_mag"], errors="coerce") < g_max]
    out = out.reset_index(drop=True)
    print(f"[herdsman] total 6D quality sample: {len(out)} stars "
          f"(d < {d_max_pc:.0f} pc, sigma_RV < {rv_err_max_kms} km/s)")
    return out


def apply_rv_zero_point(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the DR3 cool-template RV zero-point trend (Katz et al. 2023).

    For G_RVS >= 11 the published bias is
        f(G_RVS) = 0.02755 G_RVS^2 - 0.55863 G_RVS + 2.81129  km/s
    (~0 at G_RVS = 11, ~+0.4 km/s at 14, where the calibration ends); the
    correction is *subtracted*.  Beyond 14 we hold f at f(14) rather than
    extrapolate the quadratic.  A shared zero-point error is common-mode for a
    convergence (it cancels in relative motion to first order), but the trend
    is magnitude-dependent, so leaving it in would imprint a fake
    brightness-correlated radial flow.
    """
    out = df.copy()
    g = pd.to_numeric(out.get("grvs_mag"), errors="coerce").to_numpy(float)
    rv = pd.to_numeric(out["radial_velocity"], errors="coerce").to_numpy(float)
    gc = np.clip(g, None, 14.0)
    f = 0.02755 * gc ** 2 - 0.55863 * gc + 2.81129
    corr = np.where(np.isfinite(g) & (g >= 11.0), f, 0.0)
    out["radial_velocity_raw"] = rv
    out["radial_velocity"] = rv - corr
    out["rv_zeropoint_corr_kms"] = corr
    n = int((corr != 0).sum())
    print(f"[herdsman] RV zero-point correction applied to {n}/{len(out)} stars "
          f"(median {np.median(corr[corr != 0]) if n else 0.0:+.3f} km/s)")
    return out


def scalar_velocity_error(df: pd.DataFrame,
                          astro_floor_kms: float = 0.3) -> np.ndarray:
    """Per-star scalar space-velocity uncertainty (km/s).

    Quadrature sum of: the RV error plus the astrophysical line-of-sight floor
    (gravitational redshift / convective blueshift star-to-star scatter), both
    tangential proper-motion error terms, and the distance-error leverage on
    the tangential velocity.  This is the sigma that grows the meeting radius.
    """
    plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
    d_kpc = 1.0 / plx  # parallax in mas -> distance in kpc
    rv_err = pd.to_numeric(df["radial_velocity_error"], errors="coerce").to_numpy(float)
    pmra = pd.to_numeric(df["pmra"], errors="coerce").to_numpy(float)
    pmdec = pd.to_numeric(df["pmdec"], errors="coerce").to_numpy(float)
    pmra_e = pd.to_numeric(df["pmra_error"], errors="coerce").to_numpy(float)
    pmdec_e = pd.to_numeric(df["pmdec_error"], errors="coerce").to_numpy(float)
    poe = pd.to_numeric(df["parallax_over_error"], errors="coerce").to_numpy(float)

    sig_los2 = rv_err ** 2 + astro_floor_kms ** 2
    sig_tan2 = (_K * d_kpc) ** 2 * (pmra_e ** 2 + pmdec_e ** 2)
    v_tan2 = (_K * d_kpc) ** 2 * (pmra ** 2 + pmdec ** 2)
    sig_dist2 = v_tan2 / np.maximum(poe, 1.0) ** 2
    return np.sqrt(sig_los2 + sig_tan2 + sig_dist2)


__all__ = ["fetch_sample", "apply_rv_zero_point", "scalar_velocity_error"]

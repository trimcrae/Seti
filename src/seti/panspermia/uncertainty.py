"""Monte-Carlo uncertainty on the close-encounter parameters.

The encounter geometry (closest-approach distance ``d_min``, epoch ``t_enc``,
relative speed ``v_rel``) is computed from Gaia astrometry and radial velocity,
each of which carries an error.  A point estimate of "d_min = 0.9 pc" is
meaningless without its uncertainty: propagated through the parallax and
proper-motion errors, that 0.9 pc can become 0.9 +/- 3 pc.  This is why the
standard treatment of the Sun's own encounter list (Bailer-Jones 2015, 2018)
samples the 6D phase space from its covariance and reports the *distribution* of
d_min, not a single number.

Here we do the same for K2-18's encounters: resample the anchor and each candidate
from their Gaia (parallax, pmra, pmdec, radial_velocity) uncertainties, rebuild
6D phase space, recompute the linear closest approach, and report percentiles.  A
candidate whose d_min is robustly small (and t_enc robustly in the past) survives;
one whose small d_min is an astrometric fluctuation does not.

Pure / offline; the parallel point-estimate pipeline (``encounters.py``) supplies
the central values, this module supplies the error bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .encounters import _KMS_TO_PC_PER_MYR
from .kinematics import phase_space_6d

# The (value, error) astrometric/RV fields resampled per Monte-Carlo draw.
_MC_FIELDS = ("parallax", "pmra", "pmdec", "radial_velocity")


def _sample(row: dict, n: int, rng) -> pd.DataFrame:
    """Draw ``n`` Gaussian realizations of one star's astrometry + RV.

    ra/dec errors (mas) are negligible for an encounter over pc scales, so ra/dec
    are held fixed; parallax/pmra/pmdec/radial_velocity are drawn from their
    ``*_error`` columns (independent Gaussians -- Gaia correlations are a second-
    order refinement)."""
    out = {"ra": np.full(n, float(row["ra"])),
           "dec": np.full(n, float(row["dec"]))}
    for f in _MC_FIELDS:
        val = float(row.get(f, np.nan))
        err = float(row.get(f + "_error", 0.0) or 0.0)
        out[f] = rng.normal(val, max(err, 0.0), n) if np.isfinite(val) else \
            np.full(n, np.nan)
    return pd.DataFrame(out)


def mc_encounter(anchor: dict, star: dict, n: int = 2000, seed: int = 0) -> dict:
    """Monte-Carlo distribution of the encounter of ``star`` with ``anchor``.

    Both dicts carry value + ``*_error`` fields for parallax/pmra/pmdec/
    radial_velocity plus ra/dec.  Returns percentiles of d_min (pc), t_enc (Myr)
    and v_rel (km/s), plus the fraction of draws that are a *past* encounter and
    the fraction with d_min below 1 and 2 pc."""
    rng = np.random.default_rng(seed)
    a = phase_space_6d(_sample(anchor, n, rng))
    s = phase_space_6d(_sample(star, n, rng))

    r = s[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float) - a[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    dv_kms = s[["U_kms", "V_kms", "W_kms"]].to_numpy(float) - a[["U_kms", "V_kms", "W_kms"]].to_numpy(float)
    dv = dv_kms * _KMS_TO_PC_PER_MYR                       # pc/Myr
    dv2 = np.einsum("ij,ij->i", dv, dv)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_enc = np.where(dv2 > 0, -np.einsum("ij,ij->i", r, dv) / dv2, np.nan)
    d_min = np.linalg.norm(r + dv * t_enc[:, None], axis=1)
    v_rel = np.linalg.norm(dv_kms, axis=1)

    ok = np.isfinite(d_min) & np.isfinite(t_enc)
    d_min, t_enc, v_rel = d_min[ok], t_enc[ok], v_rel[ok]
    if not len(d_min):
        return {"n_valid": 0}

    def pct(x):
        return [float(np.percentile(x, p)) for p in (16, 50, 84)]

    dlo, dmed, dhi = pct(d_min)
    tlo, tmed, thi = pct(t_enc)
    vlo, vmed, vhi = pct(v_rel)
    return {
        "n_valid": int(len(d_min)),
        "d_min_p16": dlo, "d_min_p50": dmed, "d_min_p84": dhi,
        "t_enc_p16": tlo, "t_enc_p50": tmed, "t_enc_p84": thi,
        "v_rel_p16": vlo, "v_rel_p50": vmed, "v_rel_p84": vhi,
        "frac_past": float(np.mean(t_enc < 0)),
        "frac_dmin_lt1pc": float(np.mean(d_min < 1.0)),
        "frac_dmin_lt2pc": float(np.mean(d_min < 2.0)),
    }


def mc_shortlist(anchor: dict, candidates: pd.DataFrame, n: int = 2000
                 ) -> pd.DataFrame:
    """Run the Monte-Carlo encounter on every candidate row (which must carry the
    ``*_error`` columns).  Adds the percentile + fraction columns and a
    ``robust_recipient`` flag: a past encounter in the great majority of draws
    (frac_past > 0.9) that stays within 2 pc in most of them (frac_dmin_lt2pc >
    0.5).  ``seed`` is varied per row so the draws are independent."""
    rows = []
    for i, (_, c) in enumerate(candidates.iterrows()):
        m = mc_encounter(anchor, c.to_dict(), n=n, seed=1000 + i)
        rec = {**{k: c.get(k) for k in ("source_id", "ra", "dec", "dist_pc",
                                        "phot_g_mean_mag", "bp_rp",
                                        "transfer_score")}, **m}
        rec["robust_recipient"] = bool(m.get("frac_past", 0) > 0.9
                                       and m.get("frac_dmin_lt2pc", 0) > 0.5)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out) and "d_min_p50" in out:
        out = out.sort_values("d_min_p50")
    return out


_ERR_QUERY = """
SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error,
       pmdec, pmdec_error, radial_velocity, radial_velocity_error
FROM gaiadr3.gaia_source
WHERE source_id IN ({ids})
"""


def run_mc_followup(cfg=None, n: int = 3000) -> dict:
    """Runner-side follow-up: for the committed recipient shortlist, fetch the
    Gaia astrometric + RV errors (the base search kept only the values), run the
    Monte-Carlo encounter, and write ``recipient_candidates_mc.csv`` +
    ``mc_summary.json`` with robust d_min/t_enc/v_rel confidence intervals."""
    import json

    from ..config import load_config
    from .run import K2_18_SOURCE_ID, _resolve_anchor, _run_query

    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "panspermia"
    # Read source_id as int64 explicitly (a 19-digit id silently cast to float64
    # loses precision and breaks the crossmatch).
    cand = pd.read_csv(out_dir / "recipient_candidates.csv")
    cand["source_id"] = cand["source_id"].astype("int64")

    # Resolve the anchor via the base pipeline (handles the literature-RV fallback
    # when Gaia has no radial velocity for K2-18), then fetch its errors on its own
    # -- do NOT rely on it appearing in the shortlist IN-list result.
    anchor = _resolve_anchor(int(K2_18_SOURCE_ID))
    aerr = _run_query(_ERR_QUERY.format(ids=str(int(K2_18_SOURCE_ID))))
    aerr = aerr.rename(columns={c: c.lower() for c in aerr.columns})
    if len(aerr):
        ar = aerr.iloc[0].to_dict()
        for k in ("parallax_error", "pmra_error", "pmdec_error",
                  "radial_velocity_error"):
            v = ar.get(k)
            anchor[k] = float(v) if v is not None and np.isfinite(
                pd.to_numeric(v, errors="coerce")) else np.nan
    # If Gaia has no RV (hence no RV error) for the anchor, its RV came from the
    # literature; assign a conservative error so its uncertainty still propagates.
    if not np.isfinite(pd.to_numeric(anchor.get("radial_velocity_error"),
                                     errors="coerce")):
        anchor["radial_velocity_error"] = 0.5

    err = _run_query(_ERR_QUERY.format(
        ids=",".join(str(int(s)) for s in cand["source_id"])))
    err = err.rename(columns={c: c.lower() for c in err.columns})
    if len(err):
        err["source_id"] = err["source_id"].astype("int64")

    # Merge the error columns onto the shortlist (keep transfer_score/dist etc).
    ecols = ["source_id", "parallax_error", "pmra_error", "pmdec_error",
             "radial_velocity_error"]
    ecols = [c for c in ecols if c in err.columns] or ["source_id"]
    merged = cand.merge(err[ecols], on="source_id", how="left")
    mc = mc_shortlist(anchor, merged, n=n)
    mc.to_csv(out_dir / "recipient_candidates_mc.csv", index=False)

    robust = mc[mc["robust_recipient"]] if "robust_recipient" in mc else mc.iloc[:0]
    summary = {
        "n_candidates": int(len(mc)),
        "n_robust_recipients": int(len(robust)),
        "closest_median_d_min_pc": (float(mc["d_min_p50"].min())
                                    if "d_min_p50" in mc and len(mc) else None),
        "robust_source_ids": [int(s) for s in robust["source_id"]]
        if len(robust) else [],
        "top": mc.head(10).to_dict("records") if len(mc) else [],
    }
    (out_dir / "mc_summary.json").write_text(json.dumps(summary, indent=2,
                                                        default=str))
    print("[panspermia-mc]", json.dumps({k: summary[k] for k in
          ("n_candidates", "n_robust_recipients", "closest_median_d_min_pc")}))
    return summary


__all__ = ["mc_encounter", "mc_shortlist", "run_mc_followup"]

"""Empirical WISE-blend / co-movement follow-up for the WD IR-excess shortlist.

WISE confusion is the known killer of white-dwarf infrared-excess searches: the
W1/W2 beam is ~6 arcsec, while a white dwarf is intrinsically faint in the
infrared, so *any* comparably-bright red neighbour inside the beam manufactures a
fake "excess".  Project Hephaistos's main-sequence candidates fell to exactly
this (arXiv:2405.14921).

This module runs the decisive discriminator on a candidate shortlist:

  * **blend test** -- query Gaia DR3 for every neighbour within the WISE beam;
    a white dwarf is ~0 flux at W1, so a neighbour that is not far fainter than
    the WD in the red (and much brighter intrinsically, being a normal star)
    will dominate W1/W2.  We estimate each neighbour's expected W1 contribution
    from its Gaia photometry and flag the excess as a blend when a neighbour can
    supply the observed excess flux.
  * **co-movement test** -- a genuine circumstellar excess shares the white
    dwarf's (large) proper motion; a background blend does not.  A neighbour that
    is itself a high-proper-motion co-mover is physically bound (still real
    astrophysics, but not a background artefact); a zero-PM neighbour at the WD's
    position is a background star.

The maths is pure and unit-tested; the Gaia neighbour query lives in
``fetch_neighbours`` and runs on the GitHub runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_WISE_BEAM_ARCSEC = 6.5           # W1 PSF FWHM ~6.1"; use 6.5" inclusion radius
_DUST_SUBLIMATION_K = 1800.0      # silicate/carbon grains sublimate ~1500-2000 K


def _gmag_to_w1_approx(g_mag: np.ndarray, bp_rp: np.ndarray) -> np.ndarray:
    """Very rough expected W1 (Vega) for a *normal* star from Gaia G and BP-RP.

    Main-sequence stars have G - W1 that grows with colour (redder stars are
    brighter in the IR relative to G): G - W1 ~ 0.0 for blue stars up to ~4 mag
    for red M dwarfs.  A crude monotonic relation is enough to decide whether a
    neighbour can dominate the WISE beam -- we only need order-of-magnitude flux.
    """
    g = np.asarray(g_mag, float)
    c = np.asarray(bp_rp, float)
    g_minus_w1 = np.clip(0.2 + 1.1 * np.clip(c, 0.0, 4.0), 0.0, 5.0)
    return g - g_minus_w1


def _vega_mag_to_flux(mag: np.ndarray) -> np.ndarray:
    """Relative flux from a Vega magnitude (arbitrary zero-point; ratios only)."""
    return 10.0 ** (-0.4 * np.asarray(mag, float))


def blend_verdict(candidate: dict, neighbours: pd.DataFrame,
                  beam_arcsec: float = _WISE_BEAM_ARCSEC,
                  pm_comover_frac: float = 0.5) -> dict:
    """Decide whether a WD IR-excess is a WISE blend.

    ``candidate`` needs ``ra``, ``dec``, ``phot_g_mean_mag``/``g_mag`` and,
    ideally, ``pmra``/``pmdec``; ``neighbours`` is the Gaia cone around it
    (excluding the WD itself) with ``ra``,``dec``,``phot_g_mean_mag``,``bp_rp``,
    ``pmra``,``pmdec``.  Returns a verdict dict.
    """
    ra0 = float(candidate["ra"])
    dec0 = float(candidate["dec"])
    g0 = float(candidate.get("phot_g_mean_mag", candidate.get("g_mag", np.nan)))
    pmra0 = float(candidate.get("pmra", np.nan))
    pmdec0 = float(candidate.get("pmdec", np.nan))

    if neighbours is None or not len(neighbours):
        return {"verdict": "isolated", "n_beam_neighbours": 0,
                "blend_flux_ratio": 0.0, "nearest_arcsec": np.nan,
                "comoving_neighbour": False}

    nb = neighbours.copy()
    dra = (pd.to_numeric(nb["ra"], errors="coerce") - ra0) * np.cos(np.radians(dec0))
    dde = pd.to_numeric(nb["dec"], errors="coerce") - dec0
    sep = np.hypot(dra, dde) * 3600.0
    nb = nb.assign(_sep=sep)
    inbeam = nb[nb["_sep"] <= beam_arcsec]
    if not len(inbeam):
        return {"verdict": "isolated", "n_beam_neighbours": 0,
                "blend_flux_ratio": 0.0,
                "nearest_arcsec": float(np.nanmin(sep)) if len(sep) else np.nan,
                "comoving_neighbour": False}

    # Expected W1 flux of the WD itself (blue: G-W1 ~ 0) vs the neighbours'.
    wd_w1 = _vega_mag_to_flux(g0)          # WD ~ Rayleigh-Jeans, G ~ W1 -> flux(G)
    ng = pd.to_numeric(inbeam["phot_g_mean_mag"], errors="coerce").to_numpy()
    nc = pd.to_numeric(inbeam.get("bp_rp"), errors="coerce").to_numpy()
    nb_w1 = _vega_mag_to_flux(_gmag_to_w1_approx(ng, nc))
    blend_ratio = float(np.nansum(nb_w1) / (wd_w1 + 1e-30))

    # Co-moving neighbour: shares the WD's proper motion (bound companion, not a
    # background blend) -- flag separately from a static background contaminant.
    comoving = False
    if np.isfinite(pmra0) and np.isfinite(pmdec0) and (pmra0**2 + pmdec0**2) > 0:
        npmra = pd.to_numeric(inbeam.get("pmra"), errors="coerce").to_numpy()
        npmdec = pd.to_numeric(inbeam.get("pmdec"), errors="coerce").to_numpy()
        pm0 = np.hypot(pmra0, pmdec0)
        dpm = np.hypot(npmra - pmra0, npmdec - pmdec0)
        comoving = bool(np.any(dpm <= pm_comover_frac * pm0))

    # A neighbour that can supply >~10% of the WD's W1 flux ruins a clean excess.
    if blend_ratio >= 0.1:
        verdict = "comoving_blend" if comoving else "background_blend"
    else:
        verdict = "clean"
    return {"verdict": verdict, "n_beam_neighbours": int(len(inbeam)),
            "blend_flux_ratio": blend_ratio,
            "nearest_arcsec": float(inbeam["_sep"].min()),
            "comoving_neighbour": comoving}


def fetch_neighbours(ra: float, dec: float, radius_arcsec: float = 12.0,
                     self_source_id: int | None = None) -> pd.DataFrame:
    """Gaia DR3 cone around a candidate (runner-side).  Returns the FULL cone,
    including the candidate itself, so the caller can read the white dwarf's own
    G/BP-RP/PM (needed for the blend flux ratio) and then exclude it."""
    from astroquery.gaia import Gaia
    r_deg = radius_arcsec / 3600.0
    q = (f"SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, pmra, pmdec "
         f"FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec), "
         f"CIRCLE('ICRS',{ra},{dec},{r_deg}))")
    df = Gaia.launch_job_async(q).get_results().to_pandas()
    return df.rename(columns={c: c.lower() for c in df.columns}).reset_index(drop=True)


def blend_followup(candidates: pd.DataFrame, out_dir,
                   fetch=fetch_neighbours) -> dict:
    """Run the blend/co-movement test on every candidate; write a ranked table.

    A candidate that survives (``verdict='clean'`` or ``isolated``) with a large
    tau is the one worth a spectroscopic follow-up; everything else is a WISE
    blend and is retired.
    """
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, c in candidates.iterrows():
        cand = c.to_dict()
        sid = int(cand["source_id"])
        try:
            cone = fetch(float(cand["ra"]), float(cand["dec"]), self_source_id=sid)
        except Exception as exc:  # noqa: BLE001
            print(f"[blend] neighbour fetch failed for {sid}: {exc!r}")
            cone = None
        nb = None
        if cone is not None and len(cone):
            self_row = cone[cone["source_id"] == sid]
            # Fill the WD's own G/BP-RP/PM from its Gaia row (the shortlist lacks
            # them) so the blend flux ratio and co-movement test are well-defined.
            if len(self_row):
                r0 = self_row.iloc[0]
                for k in ("phot_g_mean_mag", "bp_rp", "pmra", "pmdec"):
                    if pd.isna(cand.get(k)) or cand.get(k) is None:
                        cand[k] = r0.get(k)
            nb = cone[cone["source_id"] != sid].reset_index(drop=True)
        v = blend_verdict(cand, nb)
        # Physical filter: an "excess" whose fitted dust temperature exceeds the
        # sublimation temperature cannot be circumstellar dust (or a passive dust
        # swarm) -- grains that hot vaporise.  A blackbody at 1800-3000 K with
        # order-unity tau is an *unresolved cool stellar companion* (a WD+dM/dL
        # binary, a single Gaia source, hence 'isolated'), the classic
        # WD-IR-excess contaminant.  Override a would-be-clean verdict.
        t_dust = float(cand.get("t_dust_k", np.nan))
        too_hot = np.isfinite(t_dust) and t_dust > _DUST_SUBLIMATION_K
        if too_hot and v["verdict"] in ("clean", "isolated"):
            v = {**v, "verdict": "stellar_companion", "t_dust_k": t_dust}
        rows.append({**{k: cand.get(k) for k in
                        ("source_id", "ra", "dec", "teff", "tau", "t_dust_k",
                         "multimodal_score", "simbad_id", "simbad_otype")}, **v})
    res = pd.DataFrame(rows)
    survivors = res[res["verdict"].isin(["clean", "isolated"])]
    res = res.sort_values(["verdict", "tau"], ascending=[True, False])
    res.to_csv(out_dir / "blend_followup.csv", index=False)
    summary = {
        "n_candidates": int(len(res)),
        "verdict_counts": {k: int(v) for k, v in
                           res["verdict"].value_counts().items()},
        "n_survivors": int(len(survivors)),
        "survivor_source_ids": [int(s) for s in survivors["source_id"]],
    }
    (out_dir / "blend_followup_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("[blend]", json.dumps(summary))
    return summary


__all__ = ["blend_verdict", "fetch_neighbours", "blend_followup"]

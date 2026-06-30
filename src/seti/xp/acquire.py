"""Acquire Gaia DR3 XP sampled spectra + classification metadata for a chunk.

Runs on a network-capable runner (the sandbox blocks the Gaia archive).  We pick
sources with published XP spectra in a sky cone, pull their Discrete Source
Classifier probabilities and astrophysical parameters (for the contamination
funnel), then retrieve the sampled BP/RP spectra on the common wavelength grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# XP sampled spectra are delivered on a fixed 343-point grid, 336-1020 nm.
N_SAMPLES = 343


def fetch_xp_metadata(ra: float, dec: float, radius_deg: float = 1.0,
                      g_max: float = 17.5, limit: int = 20000) -> pd.DataFrame:
    """Gaia DR3 sources with XP spectra in a cone, with classifier context."""
    from astroquery.gaia import Gaia

    q = f"""
        SELECT TOP {int(limit)} source_id, ra, dec, phot_g_mean_mag, bp_rp,
               parallax, parallax_over_error,
               classprob_dsc_combmod_quasar, classprob_dsc_combmod_galaxy,
               classprob_dsc_combmod_star, non_single_star, phot_variable_flag,
               teff_gspphot, ag_gspphot
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
          AND has_xp_sampled = 'true'
          AND phot_g_mean_mag < {g_max}
    """
    df = Gaia.launch_job_async(q).get_results().to_pandas()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    print(f"[xp] {len(df)} XP sources in cone ({ra:.2f},{dec:.2f}) r={radius_deg}")
    return df.reset_index(drop=True)


def fetch_xp_spectra(source_ids: list[int], batch: int = 5000) -> dict:
    """Retrieve XP sampled spectra for a list of source_ids.

    Returns ``{'wave': (n_wave,), 'flux': {source_id: (n_wave,) array}}``.  The
    Gaia archive caps the number of ids per ``load_data`` call, so we batch.
    """
    from astroquery.gaia import Gaia

    flux: dict[int, np.ndarray] = {}
    wave = None
    ids = [int(s) for s in source_ids]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        try:
            data = Gaia.load_data(
                ids=chunk, retrieval_type="XP_SAMPLED", data_release="Gaia DR3",
                format="csv", data_structure="INDIVIDUAL")
        except Exception as exc:
            print(f"[xp] load_data batch {i//batch} failed: {exc!r}")
            continue
        for _key, tables in (data.items() if hasattr(data, "items") else []):
            tlist = tables if isinstance(tables, list) else [tables]
            for t in tlist:
                try:
                    tdf = t.to_pandas() if hasattr(t, "to_pandas") else pd.DataFrame(t)
                except Exception:
                    continue
                cols = {c.lower(): c for c in tdf.columns}
                if "flux" not in cols:
                    continue
                sid_col = cols.get("source_id")
                w_col = cols.get("wavelength")
                if w_col is not None and wave is None:
                    wv = pd.to_numeric(tdf[w_col], errors="coerce").to_numpy()
                    if np.unique(wv).size == wv.size:        # one spectrum per table
                        wave = wv
                f = pd.to_numeric(tdf[cols["flux"]], errors="coerce").to_numpy()
                if sid_col is not None and tdf[sid_col].nunique() == 1:
                    sid = int(tdf[sid_col].iloc[0])
                    flux[sid] = f
                elif sid_col is not None:                    # stacked: group by id
                    for sid, g in tdf.groupby(sid_col):
                        flux[int(sid)] = pd.to_numeric(
                            g[cols["flux"]], errors="coerce").to_numpy()
        print(f"[xp] retrieved {len(flux)} spectra so far "
              f"({min(i+batch,len(ids))}/{len(ids)} ids)")
    if wave is None and flux:
        wave = np.arange(next(iter(flux.values())).size, dtype=float)
    return {"wave": wave, "flux": flux}


def assemble_chunk(meta: pd.DataFrame, spectra: dict) -> dict:
    """Align metadata rows with retrieved spectra into a dense matrix."""
    wave = spectra.get("wave")
    flux = spectra.get("flux", {})
    rows, mat = [], []
    n_wave = wave.size if wave is not None else 0
    for _, r in meta.iterrows():
        sid = int(r["source_id"])
        f = flux.get(sid)
        if f is None or n_wave == 0 or f.size != n_wave:
            continue
        rows.append(r)
        mat.append(f)
    if not rows:
        return {"wave": wave, "flux": np.zeros((0, n_wave)), "meta": meta.iloc[:0]}
    return {"wave": wave, "flux": np.vstack(mat),
            "meta": pd.DataFrame(rows).reset_index(drop=True)}


__all__ = ["fetch_xp_metadata", "fetch_xp_spectra", "assemble_chunk", "N_SAMPLES"]

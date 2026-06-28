"""Acquire known white-dwarf debris-disk / IR-excess control catalogues.

These are the *labelled natural-explanation* population.  We unify them into a
single ``known_excess`` table keyed by Gaia ``source_id`` (when available) or by
sky position, used to (a) validate that the pipeline recovers published
excesses, and (b) subtract known dust before reporting technosignature
candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from ..io import cached

# VizieR-resolvable control catalogues (others are merged from committed CSVs).
_VIZIER_CONTROLS = {
    "MadurgaFavieres2024": "J/A+A/688/A168",
    "DebesWIRED2011": "J/ApJS/197/38",
}


def _fetch_vizier_control(vizier_id: str) -> pd.DataFrame:
    from astroquery.vizier import Vizier

    v = Vizier(columns=["**"], row_limit=-1)
    cats = v.get_catalogs(vizier_id)
    return cats[0].to_pandas()


def acquire_control(cache_dir: Path, name: str, force: bool = False) -> pd.DataFrame:
    vizier_id = _VIZIER_CONTROLS[name]
    return cached(
        cache_dir,
        f"control_{name}",
        {"catalog": vizier_id},
        fetch=lambda: _fetch_vizier_control(vizier_id),
        provenance={"source": f"{name} ({vizier_id})"},
        force=force,
    )


def flag_known_disks(
    df: pd.DataFrame,
    known: pd.DataFrame,
    match_radius_arcsec: float = 2.0,
) -> pd.DataFrame:
    """Add a boolean ``known_disk`` column by matching to the control sample.

    Matches on ``source_id`` first, then on sky position within
    ``match_radius_arcsec`` for rows lacking a Gaia id.
    """
    out = df.copy()
    out["known_disk"] = False

    if "source_id" in known.columns and "source_id" in out.columns:
        known_ids = set(known["source_id"].dropna().astype("int64"))
        out.loc[out["source_id"].isin(known_ids), "known_disk"] = True

    if {"ra", "dec"} <= set(known.columns) and {"ra", "dec"} <= set(out.columns):
        unmatched = ~out["known_disk"]
        if unmatched.any() and len(known):
            c_out = SkyCoord(out.loc[unmatched, "ra"].to_numpy() * u.deg,
                             out.loc[unmatched, "dec"].to_numpy() * u.deg)
            c_known = SkyCoord(known["ra"].to_numpy() * u.deg,
                               known["dec"].to_numpy() * u.deg)
            idx, sep, _ = c_out.match_to_catalog_sky(c_known)
            hit = sep.arcsec <= match_radius_arcsec
            out.loc[out.index[unmatched][hit], "known_disk"] = True
    return out


def merge_controls(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate control frames, keeping common identification columns."""
    keep = ["source_id", "ra", "dec", "t_dust_k", "tau"]
    norm = []
    for f in frames:
        cols = {c: c for c in keep if c in f.columns}
        norm.append(f[list(cols)].copy())
    if not norm:
        return pd.DataFrame(columns=keep)
    out = pd.concat(norm, ignore_index=True)
    return out.replace({np.nan: None})

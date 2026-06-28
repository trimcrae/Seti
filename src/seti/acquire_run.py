"""End-to-end empirical acquisition: build the analysis-ready table from the real
catalogues, then hand it to the pipeline.

This is the single command behind ``make data``.  It pulls the Gaia EDR3
white-dwarf parent sample, its astrometric-quality columns, CatWISE2020 and
2MASS photometry, the AllWISE neighbourhood (crowding), and the control
debris-disk catalogues, then assembles them into the schema the pipeline
consumes.  All network access flows through ``seti.acquire`` (memoised to the
parquet cache), so once the cache is warm the run is offline-reproducible.

The assembly logic (:func:`assemble_analysis_table`) is a pure function with no
network dependency, so it is unit-tested offline against mock catalogue frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config

# Columns the pipeline requires, with safe defaults for optional ones.  Required
# columns (no default) must be supplied by the catalogues or assembly raises.
_OPTIONAL_DEFAULTS = {
    "W1mag_unwise": np.nan,        # unWISE confirmation: shortlist only
    "gaia_nn_arcsec": np.inf,      # nearest Gaia neighbour: no crowding penalty if absent
    "ext_flg": 0,
    "cc_flags": "0000",
    "ph_qual": "AA",
    "n_wise_neighbours": 1,
    "known_disk": False,
    "pmra_error": 1.0,
    "pmdec_error": 1.0,
    "e_pmra_wise": 50.0,
    "e_pmdec_wise": 50.0,
}

_REQUIRED = [
    "source_id", "ra", "dec", "pmra", "pmdec", "teff",
    "W1mag", "e_W1mag", "W2mag", "e_W2mag", "ra_wise", "dec_wise",
    "Jmag", "Hmag", "Ksmag",
]


def assemble_analysis_table(
    gaia_wd: pd.DataFrame,
    astrometry: pd.DataFrame,
    catwise: pd.DataFrame,
    twomass: pd.DataFrame,
    neighbourhood: pd.DataFrame | None = None,
    known: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge catalogue frames into the pipeline's analysis-ready schema.

    Joins are all on Gaia ``source_id``.  Pure / no network: every input is a
    DataFrame.  Fills optional columns with defaults and validates that the
    required columns are present and non-empty.
    """
    df = gaia_wd.copy()
    for other in (astrometry, twomass, catwise):
        if other is not None and len(other):
            cols = [c for c in other.columns if c == "source_id" or c not in df.columns]
            df = df.merge(other[cols], on="source_id", how="left")

    if neighbourhood is not None and len(neighbourhood):
        df = df.merge(neighbourhood[["source_id", "n_wise_neighbours"]],
                      on="source_id", how="left")

    # Known-disk flag from the control sample.
    df["known_disk"] = False
    if known is not None and len(known) and "source_id" in known.columns:
        known_ids = set(known["source_id"].dropna().astype("int64"))
        df.loc[df["source_id"].isin(known_ids), "known_disk"] = True

    # Derived: W2 signal-to-noise from its magnitude error.
    if "w2snr" not in df.columns and "e_W2mag" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["w2snr"] = 1.0 / (0.4 * np.log(10.0) * df["e_W2mag"])

    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"assembled table missing required columns: {missing}")
    return df


def acquire_run(
    cfg: Config | None = None,
    max_dist_pc: float | None = 100.0,
    pwd_min: float | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Acquire the real catalogues and write the analysis-ready table.

    ``max_dist_pc`` scopes the parent sample (default: the 100 pc sample, the
    forecast anchor).  ``dry_run`` validates wiring without touching the network
    by asserting the acquire functions are importable and returning an empty
    schema-correct frame.
    """
    cfg = cfg or load_config()
    cache = cfg.path("cache_dir")
    pwd_min = pwd_min if pwd_min is not None else cfg.thresholds["sample"]["pwd_min"]

    # Import here so a dry run never requires astroquery/network.
    from .acquire.controls import acquire_control, merge_controls
    from .acquire.gaia_wd import acquire_gaia_astrometry, acquire_gaia_wd
    from .acquire.twomass import acquire_twomass
    from .acquire.wise import acquire_allwise_neighbourhood, acquire_catwise

    if dry_run:
        # Verify the orchestration wiring without any network call.
        for fn in (acquire_gaia_wd, acquire_gaia_astrometry, acquire_catwise,
                   acquire_twomass, acquire_allwise_neighbourhood, acquire_control,
                   merge_controls):
            assert callable(fn)
        return pd.DataFrame(columns=_REQUIRED)

    gaia_wd = acquire_gaia_wd(cache, pwd_min=pwd_min)
    if max_dist_pc is not None and "parallax" in gaia_wd.columns:
        gaia_wd = gaia_wd[gaia_wd["parallax"] >= 1000.0 / max_dist_pc]
    if limit is not None:
        gaia_wd = gaia_wd.head(limit)
    sids = gaia_wd["source_id"].tolist()

    astrometry = acquire_gaia_astrometry(cache, sids)
    catwise = acquire_catwise(cache, gaia_wd[["source_id", "ra", "dec"]])
    twomass = acquire_twomass(cache, sids)
    neighbourhood = acquire_allwise_neighbourhood(cache, sids)

    controls = merge_controls([acquire_control(cache, "MadurgaFavieres2024"),
                               acquire_control(cache, "DebesWIRED2011")])

    table = assemble_analysis_table(gaia_wd, astrometry, catwise, twomass,
                                    neighbourhood, controls)
    out = cfg.path("processed_dir") / "analysis_ready.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    return table


def science_run(
    cfg: Config | None = None,
    max_dist_pc: float = 100.0,
    limit: int | None = None,
) -> dict:
    """Real-data run (CDS-only, runner-friendly): acquire, analyse, write results.

    Writes the analysis-ready table plus small, committable result files under
    ``results/science/`` (candidate table, summary counts, occurrence limit) so a
    CI runner can commit them back to the repository.  Returns the summary dict.
    """
    import json

    from .acquire.science import (
        fetch_catwise,
        fetch_known_disks,
        fetch_twomass,
        fetch_wd_parent,
    )
    from .pipeline import run_pipeline

    cfg = cfg or load_config()
    pwd_min = cfg.thresholds["sample"]["pwd_min"]

    parent = fetch_wd_parent(max_dist_pc, pwd_min)
    if limit is not None:
        parent = parent.head(limit)
    positions = parent[["source_id", "ra", "dec"]]

    catwise = fetch_catwise(positions)
    twomass = fetch_twomass(positions)
    known_ids = fetch_known_disks(positions)
    known = pd.DataFrame({"source_id": sorted(known_ids)}) if known_ids else None

    table = assemble_analysis_table(parent, pd.DataFrame(), catwise, twomass,
                                    neighbourhood=None, known=known)
    print(f"[science] analysis-ready table: {len(table)} white dwarfs")

    result = run_pipeline(table, cfg=cfg)

    out_dir = cfg.root / "results" / "science"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = cfg.path("processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    table.to_parquet(proc / "analysis_ready.parquet", index=False)

    cand = result.candidates
    cand_cols = [c for c in ["source_id", "ra", "dec", "teff", "t_dust_k", "tau",
                             "anomaly_score", "chi_W1", "chi_W2"] if c in cand.columns]
    cand[cand_cols].to_csv(out_dir / "candidates.csv", index=False)

    summary = {
        "max_dist_pc": max_dist_pc,
        "n_parent": int(len(table)),
        "counts": result.counts,
        "funnel_counts": result.funnel_counts,
        "occurrence_limit": result.occurrence_limit,
        "n_known_disks_matched": len(known_ids),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[science] summary:", json.dumps(summary["counts"]))
    print("[science] occurrence limit:", json.dumps(summary["occurrence_limit"]))
    return summary


__all__ = ["assemble_analysis_table", "acquire_run", "science_run"]

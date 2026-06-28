"""Offline tests for the empirical-acquisition assembly (no network)."""

import numpy as np
import pandas as pd
import pytest

from seti.acquire_run import acquire_run, assemble_analysis_table


def _mock_frames():
    gaia_wd = pd.DataFrame({
        "source_id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [0.0, 1.0, 2.0],
        "pmra": [120.0, -80.0, 200.0], "pmdec": [50.0, 30.0, -10.0],
        "parallax": [20.0, 15.0, 12.0], "teff": [10000.0, 8000.0, 12000.0],
        "logg": [8.0, 8.0, 8.0], "pwd": [0.99, 0.98, 0.97],
    })
    astrometry = pd.DataFrame({
        "source_id": [1, 2, 3], "ruwe": [1.0, 1.1, 1.05],
        "parallax_over_error": [200.0, 150.0, 120.0],
        "astrometric_excess_noise": [0.0, 0.1, 0.05],
        "pmra_error": [0.5, 0.6, 0.4], "pmdec_error": [0.5, 0.6, 0.4],
    })
    catwise = pd.DataFrame({
        "source_id": [1, 2, 3], "ra_wise": [10.0, 20.0, 30.0],
        "dec_wise": [0.0, 1.0, 2.0], "W1mag": [15.0, 16.0, 14.5],
        "e_W1mag": [0.03, 0.05, 0.03], "W2mag": [15.0, 16.0, 14.4],
        "e_W2mag": [0.04, 0.06, 0.04], "pmra_wise": [118.0, -78.0, 205.0],
        "pmdec_wise": [48.0, 31.0, -9.0], "cc_flags": ["0000", "0000", "0000"],
        "ph_qual": ["AA", "AB", "AA"],
    })
    twomass = pd.DataFrame({
        "source_id": [1, 2, 3], "Jmag": [15.2, 16.1, 14.6], "e_Jmag": [0.03] * 3,
        "Hmag": [15.3, 16.2, 14.7], "e_Hmag": [0.03] * 3,
        "Ksmag": [15.35, 16.25, 14.75], "e_Ksmag": [0.03] * 3,
    })
    neighbourhood = pd.DataFrame({"source_id": [2], "n_wise_neighbours": [3]})
    known = pd.DataFrame({"source_id": [3]})
    return gaia_wd, astrometry, catwise, twomass, neighbourhood, known


def test_assemble_produces_required_schema():
    table = assemble_analysis_table(*_mock_frames())
    for col in ("teff", "ruwe", "W1mag", "Jmag", "ra_wise", "w2snr",
                "n_wise_neighbours", "known_disk", "W1mag_unwise"):
        assert col in table.columns
    assert len(table) == 3


def test_assemble_merges_and_flags_correctly():
    table = assemble_analysis_table(*_mock_frames()).set_index("source_id")
    # Neighbourhood multiplicity merged for source 2; default 1 elsewhere.
    assert table.loc[2, "n_wise_neighbours"] == 3
    assert table.loc[1, "n_wise_neighbours"] == 1
    # Known-disk flag set only for source 3.
    assert bool(table.loc[3, "known_disk"]) is True
    assert bool(table.loc[1, "known_disk"]) is False
    # Optional columns defaulted.
    assert np.isnan(table.loc[1, "W1mag_unwise"])


def test_assemble_validates_required_columns():
    gaia_wd, astro, catwise, twomass, neigh, known = _mock_frames()
    catwise = catwise.drop(columns=["W1mag"])  # drop a required column
    with pytest.raises(ValueError, match="missing required columns"):
        assemble_analysis_table(gaia_wd, astro, catwise, twomass, neigh, known)


def test_assembled_table_runs_through_pipeline(cfg):
    # The assembled real-schema table must be accepted by the analysis pipeline.
    from seti.pipeline import run_pipeline

    table = assemble_analysis_table(*_mock_frames())
    result = run_pipeline(table, cfg=cfg)
    assert result.counts["parent"] == 3
    assert "f_upper" in result.occurrence_limit


def test_acquire_run_dry_run_no_network(cfg):
    # Dry run validates wiring and returns a schema-correct empty frame.
    table = acquire_run(cfg, dry_run=True)
    assert len(table) == 0
    assert "W1mag" in table.columns

"""Tests for the emission-line-galaxy rejection stage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from seti.spectra.galaxy_reject import (
    GALAXY_LINES,
    flag_galaxy_spectra,
    galaxy_redshift_match,
)


def test_halpha_nii_pair_is_flagged_galaxy():
    # Halpha + [N II] 6584 redshifted to z=0.1452 -> the real 7518/7542 pair.
    z = 0.1452
    waves = [GALAXY_LINES["Ha6563"] * (1 + z), GALAXY_LINES["[NII]6584"] * (1 + z)]
    r = galaxy_redshift_match(waves)
    assert r["is_galaxy"] is True
    assert abs(r["z"] - z) < 1e-3
    assert set(r["rest_lines"]) == {"Ha6563", "[NII]6584"}


def test_oiii_doublet_is_flagged_galaxy():
    z = 0.3
    waves = [GALAXY_LINES["[OIII]4959"] * (1 + z),
             GALAXY_LINES["[OIII]5007"] * (1 + z)]
    assert galaxy_redshift_match(waves)["is_galaxy"] is True


def test_unrelated_line_pair_not_flagged():
    # Two grid lines that are NOT a locked diagnostic pair, at a contrived z --
    # must NOT be called a galaxy (this is the variable-star false-positive guard).
    z = 0.05
    waves = [GALAXY_LINES["Hb4862"] * (1 + z) * 1.001,      # off by ~300 km/s
             GALAXY_LINES["[NeIII]3869"] * (1 + z)]
    assert galaxy_redshift_match(waves)["is_galaxy"] is False


def test_single_line_cannot_be_galaxy():
    assert galaxy_redshift_match([7518.0])["is_galaxy"] is False


def test_flag_galaxy_spectra_marks_group():
    z = 0.1452
    df = pd.DataFrame({
        "spec_id": ["A", "A", "B"],
        "wavelength": [GALAXY_LINES["Ha6563"] * (1 + z),
                       GALAXY_LINES["[NII]6584"] * (1 + z),
                       7000.0],
    })
    out = flag_galaxy_spectra(df)
    assert out.loc[out["spec_id"] == "A", "is_galaxy"].all()
    assert not out.loc[out["spec_id"] == "B", "is_galaxy"].any()
    assert np.isclose(out.loc[out["spec_id"] == "A", "galaxy_z"].iloc[0], z, atol=1e-3)

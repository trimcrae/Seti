"""Synthetic production-path VSX witness; no live catalogue claim."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from seti.parallax4 import deepvet as DV


def neighboring_eclipsing_row():
    return {
        "OID": "9007199254740993", "Name": "synthetic-neighbor", "Type": "EA",
        "Period": 1.2, "RAJ2000": 10 + 9 / 3600 / np.cos(np.deg2rad(-5)),
        "DEJ2000": -5.0, "Epoch": 2450000.0,
    }


def test_baseline_vsx_neighbor_becomes_target_mechanism(monkeypatch):
    rows = pd.DataFrame([neighboring_eclipsing_row()])
    monkeypatch.setattr(DV, "_tap_rows", lambda *a, **kw: rows)
    result = DV.vsx_type(10, -5)
    fate, _ = DV.classify_fate(result)
    target = SkyCoord(10 * u.deg, -5 * u.deg)
    neighbor = SkyCoord(rows.iloc[0]["RAJ2000"] * u.deg, -5 * u.deg)
    sep = target.separation(neighbor).arcsec
    assert sep == pytest.approx(9.0, abs=1e-5)
    assert result["vsx_type"] == "EA"
    assert fate == "KNOWN_ECLIPSING_BINARY(VSX:EA)"
    print(f"BASELINE_VSX_WITNESS sep_arcsec={sep:.9f} fate={fate}")

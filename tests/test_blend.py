"""Tests for the WISE-blend / co-movement WD IR-excess follow-up."""
from __future__ import annotations

import numpy as np
import pandas as pd

from seti.discriminate.blend import blend_followup, blend_verdict


def _wd():
    # A blue, high-proper-motion white dwarf: faint at W1.
    return {"source_id": 1, "ra": 100.0, "dec": 20.0, "phot_g_mean_mag": 16.0,
            "bp_rp": 0.0, "pmra": 200.0, "pmdec": -150.0, "tau": 0.6}


def test_isolated_wd_is_clean():
    nb = pd.DataFrame(columns=["ra", "dec", "phot_g_mean_mag", "bp_rp", "pmra", "pmdec"])
    v = blend_verdict(_wd(), nb)
    assert v["verdict"] == "isolated"
    assert v["n_beam_neighbours"] == 0


def test_bright_red_neighbour_in_beam_is_blend():
    # A red neighbour 3" away, comparable G but much brighter in W1 -> dominates.
    nb = pd.DataFrame({
        "ra": [100.0 + 3.0 / 3600.0 / np.cos(np.radians(20.0))],
        "dec": [20.0], "phot_g_mean_mag": [16.5], "bp_rp": [3.0],
        "pmra": [0.0], "pmdec": [0.0]})       # zero PM -> background
    v = blend_verdict(_wd(), nb)
    assert v["verdict"] == "background_blend"
    assert v["blend_flux_ratio"] > 0.1
    assert v["comoving_neighbour"] is False


def test_faint_distant_neighbour_stays_clean():
    # A neighbour outside the beam (10") does not contaminate.
    nb = pd.DataFrame({
        "ra": [100.0 + 10.0 / 3600.0 / np.cos(np.radians(20.0))],
        "dec": [20.0], "phot_g_mean_mag": [15.0], "bp_rp": [3.0],
        "pmra": [0.0], "pmdec": [0.0]})
    v = blend_verdict(_wd(), nb)
    assert v["verdict"] in ("clean", "isolated")
    assert v["n_beam_neighbours"] == 0


def test_comoving_neighbour_flagged():
    # A neighbour in the beam sharing the WD's proper motion = bound, not background.
    nb = pd.DataFrame({
        "ra": [100.0 + 2.0 / 3600.0 / np.cos(np.radians(20.0))],
        "dec": [20.0], "phot_g_mean_mag": [16.0], "bp_rp": [3.0],
        "pmra": [205.0], "pmdec": [-148.0]})
    v = blend_verdict(_wd(), nb)
    assert v["comoving_neighbour"] is True
    assert v["verdict"] == "comoving_blend"


def test_blend_followup_offline():
    cands = pd.DataFrame([{"source_id": 1, "ra": 100.0, "dec": 20.0, "teff": 6000,
                           "tau": 0.6, "t_dust_k": 2000, "multimodal_score": 0.3,
                           "phot_g_mean_mag": 16.0, "bp_rp": 0.0,
                           "pmra": 200.0, "pmdec": -150.0,
                           "simbad_id": None, "simbad_otype": None}])
    # Fake fetch returns the FULL cone (self + one bright red static neighbour).
    def fake(ra, dec, self_source_id=None, radius_arcsec=12.0):
        return pd.DataFrame({
            "source_id": [1, 999],
            "ra": [ra, ra + 2.0 / 3600.0], "dec": [dec, dec],
            "phot_g_mean_mag": [16.0, 16.0], "bp_rp": [0.0, 3.2],
            "pmra": [200.0, 0.0], "pmdec": [-150.0, 0.0]})
    import tempfile
    d = tempfile.mkdtemp()
    s = blend_followup(cands, d, fetch=fake)
    assert s["n_candidates"] == 1
    assert s["n_survivors"] == 0

"""Tests for the contamination funnel, including the novel co-movement cut."""

import pandas as pd

from seti.contamination import run_funnel
from seti.contamination.comovement import comovement_pass, propagated_offset_arcsec


def test_funnel_rejects_contaminants_keeps_clean(cfg, sample):
    sample = sample.copy()
    sample.attrs["gaia_ref_epoch"] = 2016.0
    sample.attrs["wise_mean_epoch"] = 2010.5
    vetted = run_funnel(sample, cfg.thresholds)

    # Clean and astrophysically-real (disk/anomaly) sources should survive.
    for label in ("clean", "known_disk", "anomaly"):
        keep = vetted.loc[vetted.label == label, "clean"].mean()
        assert keep > 0.7, f"{label} survival too low: {keep}"

    # Designed contaminants should be overwhelmingly rejected.
    for label in ("blend", "background", "agn"):
        keep = vetted.loc[vetted.label == label, "clean"].mean()
        assert keep < 0.2, f"{label} leaked through: {keep}"


def test_funnel_counts_monotonic_nonincreasing(cfg, sample):
    vetted = run_funnel(sample, cfg.thresholds)
    counts = list(vetted.attrs["funnel_counts"].values())
    assert all(b <= a for a, b in zip(counts, counts[1:], strict=False))


def test_comovement_offset_zero_for_comoving_source():
    # A source whose WISE position equals the epoch-propagated Gaia position has
    # ~zero offset; a static (non-co-moving) source at the Gaia epoch does not.
    dt_deg = 100.0 * (2010.5 - 2016.0) / 3.6e6  # pmra=100 mas/yr over the epoch gap
    df = pd.DataFrame({
        "ra": [150.0, 150.0], "dec": [0.0, 0.0],
        "pmra": [100.0, 100.0], "pmdec": [0.0, 0.0],
        "ra_wise": [150.0 + dt_deg, 150.0],  # co-moving, then static
        "dec_wise": [0.0, 0.0],
    })
    off = propagated_offset_arcsec(df, 2016.0, 2010.5)
    assert off.iloc[0] < 0.05          # co-moving -> tiny offset
    assert off.iloc[1] > 0.4           # static -> large offset (~0.55")


def test_comovement_pass_rejects_static(cfg):
    # High proper motion so the static source's epoch-propagated offset exceeds
    # the 1" position threshold; CatWISE PMs included so the PM sub-test also bites.
    pm = 300.0
    dt_deg = pm * (2010.5 - 2016.0) / 3.6e6
    df = pd.DataFrame({
        "ra": [150.0, 150.0], "dec": [0.0, 0.0],
        "pmra": [pm, pm], "pmdec": [0.0, 0.0],
        "pmra_error": [1.0, 1.0], "pmdec_error": [1.0, 1.0],
        "pmra_wise": [pm, 0.0], "pmdec_wise": [0.0, 0.0],   # co-moving, then static
        "e_pmra_wise": [15.0, 15.0], "e_pmdec_wise": [15.0, 15.0],
        "ra_wise": [150.0 + dt_deg, 150.0],
        "dec_wise": [0.0, 0.0],
    })
    df.attrs["gaia_ref_epoch"] = 2016.0
    df.attrs["wise_mean_epoch"] = 2010.5
    out = comovement_pass(df, cfg.thresholds)
    assert bool(out.iloc[0]) is True
    assert bool(out.iloc[1]) is False

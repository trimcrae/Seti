"""Tests for the quantitative co-movement contamination budget."""

import numpy as np

from seti.population import generate_population
from seti.stats.contamination_budget import (
    chance_alignment_rate,
    contamination_budget,
    efficacy_vs_pm,
)


def test_chance_alignment_rate_scaling():
    # Doubling the radius quadruples the chance-alignment area/rate.
    r1 = chance_alignment_rate(46000.0, 3.0)
    r2 = chance_alignment_rate(46000.0, 6.0)
    assert np.isclose(r2 / r1, 4.0)
    # At CatWISE density within 3", the rate is a few to ~ten per cent.
    assert 0.02 < r1 < 0.2


def test_budget_reduces_contamination(cfg):
    pop = generate_population(cfg, seed=3)
    b = contamination_budget(cfg, pop)
    # The cut removes a substantial fraction and never increases contamination.
    assert 0.3 < b["removed_fraction"] < 1.0
    assert b["chance_aligned_after_real"] < b["chance_aligned_before_real"]
    assert b["rejection_factor"] > 1.0


def test_efficacy_increases_with_proper_motion(cfg):
    pop = generate_population(cfg, seed=3)
    eff = efficacy_vs_pm(cfg, pop).sort_values("pm_mid")
    rem = eff["removed_fraction"].to_numpy()
    # Removal efficacy is monotonically non-decreasing with proper motion and
    # reaches ~1 for the fastest-moving white dwarfs.
    assert np.all(np.diff(rem) >= -1e-9)
    assert rem[-1] > 0.95

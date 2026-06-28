"""End-to-end pipeline tests on the offline synthetic sample."""

import numpy as np

from seti.pipeline import run_pipeline
from seti.stats.completeness import completeness_map


def test_pipeline_candidates_are_anomalies(cfg, sample):
    result = run_pipeline(sample, cfg=cfg)
    cand = result.candidates
    # Every surviving candidate must be an injected anomaly -- no contaminant or
    # known disk should leak into the candidate list.
    assert len(cand) >= 1
    assert set(cand["label"].unique()) <= {"anomaly"}


def test_pipeline_subtracts_known_disks(cfg, sample):
    result = run_pipeline(sample, cfg=cfg)
    # Known debris disks are flagged and excluded from candidates.
    assert result.counts["known_disk"] >= 1
    assert (result.candidates["known_disk"] == False).all()  # noqa: E712


def test_pipeline_reports_occurrence_limit(cfg, sample):
    result = run_pipeline(sample, cfg=cfg)
    occ = result.occurrence_limit
    assert occ["n_eff"] > 0
    assert occ["f_upper"] > occ["f_point"]  # upper limit above the point estimate


def test_completeness_map_monotonic_in_tau(cfg, sample):
    clean = sample[sample.label == "clean"]
    cmap = completeness_map(clean, cfg.thresholds,
                            t_grid=[300, 1200], tau_grid=[0.003, 0.1])
    # At fixed temperature, higher tau (stronger excess) must be recovered at
    # least as often as lower tau.
    for t in (300, 1200):
        sub = cmap[cmap.t_dust_k == t].sort_values("tau")
        rec = sub["recovered_fraction"].to_numpy()
        assert np.all(np.diff(rec) >= -1e-9)

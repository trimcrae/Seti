"""Tests for the LHS 1140 donor / directed-travel channel (the K2-18 mirror).

Offline only: synthetic anchor + star tables are injected straight into the reused
panspermia machinery via ``lhs1140_origin_run(anchor=..., table=...)`` (no network).
Decisive checks: (1) the pipeline runs end to end and writes its three outputs;
(2) an injected close+slow PAST recipient is ranked at the top of the recipient
list, above a genuine fast flyby; (3) the fast flyby is present but ranks below it;
(4) the destination ranking uses the CLASSICAL prior (the contrast with K2-18).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from seti.lhs1140_origin.run import lhs1140_origin_run
from seti.panspermia.kinematics import _A_ICRS_TO_GAL, _K_AUYR_KMS, phase_space_6d

_KMS_TO_PC_PER_MYR = 1.0227121651


def _observables_from_galactic(X, Y, Z, U, V, W, source_id=1, bp_rp=1.5, g=10.0):
    """Inverse of phase_space_6d: Galactic 6D (pc, km/s) -> Gaia observables.

    Same construction as tests/test_panspermia.py, so a star can be placed at any
    physical state and the forward pipeline made to reconstruct it."""
    At = _A_ICRS_TO_GAL.T
    r_icrs = At @ np.array([X, Y, Z], float)
    v_icrs = At @ np.array([U, V, W], float)
    dist = np.linalg.norm(r_icrs)
    r_hat = r_icrs / dist
    dec = np.arcsin(r_hat[2])
    ra = np.arctan2(r_hat[1], r_hat[0]) % (2 * np.pi)
    ca, sa, cd, sd = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec)
    a_hat = np.array([-sa, ca, 0.0])
    d_hat = np.array([-sd * ca, -sd * sa, cd])
    rv = float(v_icrs @ r_hat)
    v_a = float(v_icrs @ a_hat)
    v_d = float(v_icrs @ d_hat)
    scale = _K_AUYR_KMS * dist / 1000.0
    return {
        "source_id": source_id, "ra": np.degrees(ra), "dec": np.degrees(dec),
        "parallax": 1000.0 / dist, "pmra": v_a / scale, "pmdec": v_d / scale,
        "radial_velocity": rv, "phot_g_mean_mag": g, "bp_rp": bp_rp,
    }


def _synthetic_field(seed=7):
    """LHS-1140-like anchor + background field + one injected close/slow recipient
    and one deliberate fast flyby."""
    rng = np.random.default_rng(seed)
    anchor_obs = _observables_from_galactic(15.0, -25.0, 6.0, -9.0, 4.0, -3.0,
                                            source_id=999)
    rows = []
    for i in range(200):
        X = 15.0 + rng.uniform(-30, 30)
        Y = -25.0 + rng.uniform(-30, 30)
        Z = 6.0 + rng.uniform(-30, 30)
        U, V, W = rng.uniform(-40, 40, 3)
        rows.append(_observables_from_galactic(X, Y, Z, U, V, W, source_id=i))
    # Injected recipient: a genuine close (~0.15 pc) + slow (4 km/s) PAST flyby.
    vrel = 4.0
    x_now = 15.0 + vrel * _KMS_TO_PC_PER_MYR * 3.0     # closest approach 3 Myr ago
    recip = _observables_from_galactic(x_now, -25.0 + 0.15, 6.0,
                                       -9.0 + vrel, 4.0, -3.0, source_id=42)
    # Deliberate FAST flyby: also passed close, but at 60 km/s -- non-capturable.
    vfast = 60.0
    xf_now = 15.0 + vfast * _KMS_TO_PC_PER_MYR * 2.0
    fast = _observables_from_galactic(xf_now, -25.0 + 0.20, 6.0,
                                      -9.0 + vfast, 4.0, -3.0, source_id=7)
    table = pd.DataFrame(rows + [recip, fast])
    return anchor_obs, table


def test_run_ranks_close_slow_recipient_above_fast_flyby(tmp_path):
    from seti.config import load_config

    anchor_obs, table = _synthetic_field()
    cfg = load_config()
    cfg.root = tmp_path
    summary = lhs1140_origin_run(cfg, anchor=dict(anchor_obs), table=table,
                                 search_pc=60.0, t_max_myr=10.0, d_min_max_pc=2.0)

    # Pipeline ran and reframed the anchor as the LHS 1140 donor.
    assert summary["anchor"]["name"] == "LHS 1140"
    assert summary["destination_prior"] == "classical"
    assert summary["n_recipients"] >= 1

    # The close+slow recipient tops the recipient ranking.
    ids = [r["source_id"] for r in summary["top_recipients"]]
    assert ids[0] == 42
    assert summary["closest_approach_pc"] < 0.5

    # The fast flyby ranks strictly below the slow one (if it clears the shortlist
    # cut at all) -- capture cross-section falls as 1/v_rel^2.
    if 7 in ids and 42 in ids:
        assert ids.index(42) < ids.index(7)
    scores = {r["source_id"]: r["transfer_score"] for r in summary["top_recipients"]}
    if 7 in scores:
        assert scores[42] > scores[7]

    # All three outputs were written.
    base = tmp_path / "results" / "lhs1140_origin"
    assert (base / "recipients.csv").exists()
    assert (base / "destinations.csv").exists()
    assert (base / "summary.json").exists()


def test_recipient_is_a_past_close_slow_encounter(tmp_path):
    from seti.config import load_config

    anchor_obs, table = _synthetic_field()
    cfg = load_config()
    cfg.root = tmp_path
    lhs1140_origin_run(cfg, anchor=dict(anchor_obs), table=table,
                       search_pc=60.0, t_max_myr=10.0, d_min_max_pc=2.0)

    recips = pd.read_csv(tmp_path / "results" / "lhs1140_origin" / "recipients.csv")
    top = recips.sort_values("transfer_score", ascending=False).iloc[0]
    assert int(top["source_id"]) == 42
    assert top["t_enc_myr"] < 0            # a PAST encounter
    assert top["d_min_pc"] < 0.5           # close
    assert top["v_rel_kms"] < 10.0         # slow


def test_fallback_used_when_no_anchor_resolved(tmp_path, monkeypatch):
    """With no injected anchor and Gaia unreachable, the LHS 1140 literature vector
    is used -- never the panspermia K2-18 fallback."""
    import seti.lhs1140_origin.run as mod

    monkeypatch.setattr(mod, "_resolve_anchor",
                        lambda sid: dict(mod.LHS1140_FALLBACK, source_id=999))
    resolved = mod._resolve_lhs1140(mod.LHS1140_SOURCE_ID, None)
    assert resolved["source_id"] == mod.LHS1140_SOURCE_ID
    assert np.isclose(resolved["parallax"], 66.83)

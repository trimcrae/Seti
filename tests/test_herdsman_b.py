"""HERDSMAN-B offline tests: scorer physics + synthetic-census injection.

The injection test builds a census of 150 honestly co-natal clusters (with
realistic per-star errors, interlopers, and Teff systematics) plus one
"assembly" whose members sample the field metallicity distribution, and
requires the scorer to flag exactly the assembly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seti.herdsman_b.score import (
    _field_spread_at,
    build_field_spread,
    detrend_teff,
    radec_to_rgal,
    score_census,
)
from seti.panspermia.kinematics import _A_ICRS_TO_GAL


def _radec_of_galactic_unit(unit):
    """Sky position (deg) of a heliocentric Galactic unit vector."""
    r_icrs = _A_ICRS_TO_GAL.T @ np.asarray(unit, float)
    dec = np.degrees(np.arcsin(r_icrs[2]))
    ra = np.degrees(np.arctan2(r_icrs[1], r_icrs[0])) % 360.0
    return ra, dec


def test_radec_to_rgal_geometry():
    ra_gc, dec_gc = _radec_of_galactic_unit([1.0, 0.0, 0.0])   # toward GC
    assert abs(radec_to_rgal(ra_gc, dec_gc, 8.178)) < 0.05     # lands at GC
    assert abs(radec_to_rgal(ra_gc, dec_gc, 4.0) - 4.178) < 0.05
    ra_ac, dec_ac = _radec_of_galactic_unit([-1.0, 0.0, 0.0])  # anti-centre
    assert abs(radec_to_rgal(ra_ac, dec_ac, 2.0) - 10.178) < 0.05


def test_detrend_removes_slope_keeps_real_spread():
    rng = np.random.default_rng(5)
    teff = rng.uniform(4500, 7000, 200)
    slope = 1e-4                                  # dex/K systematic
    mh_sys = slope * (teff - 5750) + rng.normal(0, 0.05, 200)
    r = detrend_teff(mh_sys, teff)
    assert np.std(r) < 0.07                       # trend removed
    mh_real = rng.normal(0, 0.2, 200) + rng.normal(0, 0.05, 200)
    r2 = detrend_teff(mh_real, teff)
    assert np.std(r2) > 0.17                      # true spread survives


def test_field_spread_bins():
    rng = np.random.default_rng(7)
    field = pd.DataFrame({"r_gal": rng.uniform(6, 10, 20000),
                          "mh": rng.normal(-0.1, 0.22, 20000)})
    fs = build_field_spread(field)          # no dist_kpc -> marginal only
    filled = np.isfinite(fs.spreads_r)
    assert filled.sum() >= 3
    assert np.all(np.abs(fs.spreads_r[filled] - 0.22) < 0.03)
    # Without distance information every 2D bin is empty and lookups fall
    # back to the R_gal marginal.
    assert not np.isfinite(fs.spreads_2d).any()
    s, matched = _field_spread_at(fs, 8.0, 1.0)
    assert matched is False and abs(s - 0.22) < 0.03


def test_field_spread_distance_matched_and_fallback():
    rng = np.random.default_rng(9)
    # Same R_gal everywhere; nearby field is tight (0.10 dex), the distant
    # field is inflated (0.40 dex) — the v1 systematic in miniature.
    near = pd.DataFrame({"r_gal": rng.uniform(7.6, 8.4, 8000),
                         "dist_kpc": rng.uniform(0.1, 0.45, 8000),
                         "mh": rng.normal(-0.1, 0.10, 8000)})
    far = pd.DataFrame({"r_gal": rng.uniform(7.6, 8.4, 8000),
                        "dist_kpc": rng.uniform(2.05, 2.45, 8000),
                        "mh": rng.normal(-0.1, 0.40, 8000)})
    fs = build_field_spread(pd.concat([near, far], ignore_index=True))
    s_near, m_near = _field_spread_at(fs, 8.0, 0.3)
    s_far, m_far = _field_spread_at(fs, 8.0, 2.2)
    assert m_near is True and abs(s_near - 0.10) < 0.02
    assert m_far is True and abs(s_far - 0.40) < 0.05
    # A distance bin with no stars falls back to the R_gal marginal.
    s_fb, m_fb = _field_spread_at(fs, 8.0, 7.0)
    assert m_fb is False and np.isfinite(s_fb)


def _synthetic_census(seed=11, n_conatal=150):
    rng = np.random.default_rng(seed)
    rows = []

    def add_cluster(name, mh_values, sig_values, probs, teff, dist=None,
                    gmag=None):
        n = len(mh_values)
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        # Inside the v2 GSP-Phot trust region (< 2.5 kpc) unless overridden.
        dist = rng.uniform(0.5, 2.2) if dist is None else dist
        ra, dec = _radec_of_galactic_unit(u)
        gmag = np.full(n, 14.0) if gmag is None else np.asarray(gmag, float)
        for j in range(n):
            rows.append({"cluster": name, "source_id": len(rows),
                         "prob": probs[j], "mh": mh_values[j],
                         "mh_sigma": sig_values[j], "teff": teff[j],
                         "gmag": gmag[j], "ra": ra, "dec": dec,
                         "parallax": 1.0 / dist})

    for k in range(n_conatal):
        n = int(rng.integers(12, 80))
        mu = rng.normal(-0.10, 0.15)
        teff = rng.uniform(4500, 7200, n)
        sig = rng.uniform(0.06, 0.12, n)
        mh = mu + rng.normal(0, 0.03, n) + rng.normal(0, 1, n) * sig \
            + 6e-5 * (teff - 5800)                   # shared Teff systematic
        probs = rng.uniform(0.90, 1.0, n)
        n_int = max(1, int(0.06 * n))                # field interlopers
        idx = rng.choice(n, n_int, replace=False)
        mh[idx] = rng.normal(-0.10, 0.22, n_int)
        probs[idx] = rng.uniform(0.70, 0.90, n_int)
        add_cluster(f"conatal_{k:03d}", mh, sig, probs, teff)

    n = 45                                            # the gathered assembly
    teff = rng.uniform(4500, 7200, n)
    sig = rng.uniform(0.06, 0.12, n)
    mh = rng.normal(-0.10, 0.20, n) + rng.normal(0, 1, n) * sig
    add_cluster("assembly", mh, sig, rng.uniform(0.9, 1.0, n), teff)

    members = pd.DataFrame(rows)
    field = pd.DataFrame({"r_gal": rng.uniform(4, 12, 30000),
                          "dist_kpc": rng.uniform(0.3, 2.4, 30000),
                          "mh": rng.normal(-0.1, 0.22, 30000)})
    return members, field


def test_synthetic_census_flags_only_the_assembly():
    members, field = _synthetic_census()
    tab = score_census(members, field)
    assert len(tab) > 140
    top = tab.iloc[0]
    assert top["cluster"] == "assembly"
    assert bool(top["assembly_candidate"])
    assert top["x_trim"] >= 2.0 and top["z_census"] >= 4.0
    false_pos = tab[(tab["assembly_candidate"]) & (tab["cluster"] != "assembly")]
    assert len(false_pos) == 0, false_pos["cluster"].tolist()


def test_two_population_mimic_is_flagged_not_candidate():
    members, field = _synthetic_census(seed=12, n_conatal=60)
    rng = np.random.default_rng(3)
    n = 40                        # stripped-nucleus mimic: two tight [M/H] peaks
    teff = rng.uniform(4500, 7200, n)
    sig = rng.uniform(0.06, 0.10, n)
    mh = np.where(rng.random(n) < 0.5, -0.55, 0.05) + rng.normal(0, 0.04, n)
    rows = [{"cluster": "nucleus", "source_id": 10_000_000 + j,
             "prob": 0.95, "mh": mh[j], "mh_sigma": sig[j], "teff": teff[j],
             "gmag": 14.0, "ra": 120.0, "dec": -30.0, "parallax": 0.5}
            for j in range(n)]
    members = pd.concat([members, pd.DataFrame(rows)], ignore_index=True)
    tab = score_census(members, field)
    row = tab[tab["cluster"] == "nucleus"].iloc[0]
    assert bool(row["two_pop"]) is True
    assert bool(row["assembly_candidate"]) is False


def test_mag_systematic_veto():
    """A Hogg_4-style mh-vs-G trend is flagged and vetoed (v2 kill switch)."""
    members, field = _synthetic_census(seed=31, n_conatal=60)
    rng = np.random.default_rng(4)
    n = 60
    gmag = rng.uniform(12.0, 17.0, n)
    teff = rng.uniform(4500, 7200, n)             # uncorrelated with gmag
    sig = rng.uniform(0.06, 0.10, n)
    mh = -0.10 - 0.25 * (gmag - 14.5) + rng.normal(0, 0.05, n)
    rows = [{"cluster": "magsys", "source_id": 20_000_000 + j,
             "prob": 0.95, "mh": mh[j], "mh_sigma": sig[j], "teff": teff[j],
             "gmag": gmag[j], "ra": 150.0, "dec": -40.0, "parallax": 1 / 1.5}
            for j in range(n)]
    members = pd.concat([members, pd.DataFrame(rows)], ignore_index=True)
    tab = score_census(members, field)
    row = tab[tab["cluster"] == "magsys"].iloc[0]
    assert row["s_trim"] > 0.15                   # spread big enough to tempt
    assert abs(row["corr_mh_gmag"]) > 0.4
    assert bool(row["mag_systematic"]) is True
    assert bool(row["assembly_candidate"]) is False
    # Constant-G clusters must have an undefined correlation, not a veto.
    others = tab[tab["cluster"] != "magsys"]
    assert not others["mag_systematic"].any()


def test_beyond_phot_trust_veto():
    """A field-like spread at 3.5 kpc is scored but flagged, never a candidate."""
    members, field = _synthetic_census(seed=32, n_conatal=60)
    rng = np.random.default_rng(6)
    n = 45
    teff = rng.uniform(4500, 7200, n)
    sig = rng.uniform(0.06, 0.12, n)
    mh = rng.normal(-0.10, 0.20, n) + rng.normal(0, 1, n) * sig
    rows = [{"cluster": "far_assembly", "source_id": 30_000_000 + j,
             "prob": 0.95, "mh": mh[j], "mh_sigma": sig[j], "teff": teff[j],
             "gmag": 14.0, "ra": 80.0, "dec": 10.0, "parallax": 1 / 3.5}
            for j in range(n)]
    members = pd.concat([members, pd.DataFrame(rows)], ignore_index=True)
    tab = score_census(members, field)
    row = tab[tab["cluster"] == "far_assembly"].iloc[0]
    assert 3.4 < row["dist_kpc"] < 3.6
    assert bool(row["beyond_phot_trust"]) is True
    assert bool(row["gc_like"]) is False          # < 4 kpc, not metal-poor
    assert bool(row["assembly_candidate"]) is False
    near = tab[tab["cluster"] != "far_assembly"]
    assert not near["beyond_phot_trust"].any()


def test_gc_like_veto():
    """Metal-poor rich clusters and very distant ones carry gc_like."""
    members, field = _synthetic_census(seed=33, n_conatal=60)
    rng = np.random.default_rng(8)

    def bulk(name, n, mu, dist, base):
        teff = rng.uniform(4500, 7200, n)
        sig = rng.uniform(0.06, 0.12, n)
        mh = mu + rng.normal(0, 0.05, n)
        return [{"cluster": name, "source_id": base + j, "prob": 0.95,
                 "mh": mh[j], "mh_sigma": sig[j], "teff": teff[j],
                 "gmag": 14.0, "ra": 200.0, "dec": -50.0,
                 "parallax": 1.0 / dist} for j in range(n)]

    rows = bulk("globular", 150, -1.2, 2.0, 40_000_000) \
        + bulk("distant", 40, -0.1, 5.0, 41_000_000)
    members = pd.concat([members, pd.DataFrame(rows)], ignore_index=True)
    tab = score_census(members, field)
    gc = tab[tab["cluster"] == "globular"].iloc[0]
    far = tab[tab["cluster"] == "distant"].iloc[0]
    assert bool(gc["gc_like"]) is True            # metal-poor and populous
    assert bool(far["gc_like"]) is True           # beyond 4 kpc
    assert bool(gc["assembly_candidate"]) is False
    assert bool(far["assembly_candidate"]) is False
    assert not tab[~tab["cluster"].isin(["globular", "distant"])
                   ]["gc_like"].any()


def test_score_and_write_outputs(tmp_path):
    from seti.config import load_config
    from seti.herdsman_b.run import score_and_write

    members, field = _synthetic_census(seed=21, n_conatal=40)
    cfg = load_config()
    cfg.root = tmp_path
    summary = score_and_write(cfg, members, field)
    out = tmp_path / "results" / "herdsman_b"
    assert (out / "cluster_scores.csv").exists()
    assert (out / "candidates.json").exists()
    assert (out / "REPORT.md").exists()
    assert summary["n_clusters_scored"] >= 35

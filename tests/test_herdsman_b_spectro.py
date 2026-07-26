"""HERDSMAN-B v2 spectroscopic-crossmatch offline tests.

Synthetic GALAH-like census: ~40 honestly co-natal clusters with realistic
per-star [Fe/H] errors (0.05-0.1 dex) plus one injected field-sampled
assembly; the scorer must rank the assembly first, flag it, and produce no
false positives. The quality-flag filter and the >= 6-member requirement
are exercised on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seti.herdsman_b.spectro import (
    SPECTRO_QUALITY,
    resolve_spectro_columns,
    score_spectro_census,
    standardize_spectro,
)


def _synthetic_spectro(seed=23, n_conatal=40):
    """(members, spectro) frames in the standardized offline schema."""
    rng = np.random.default_rng(seed)
    mem_rows, spec_rows = [], []
    sid = [1_000_000]

    def add_cluster(name, fe_values, e_values, probs):
        for fe, e, p in zip(fe_values, e_values, probs, strict=True):
            sid[0] += 1
            mem_rows.append({"source_id": sid[0], "cluster": name, "prob": p})
            spec_rows.append({"source_id": sid[0], "fe_h": fe, "e_fe_h": e,
                              "r_gal": np.nan, "dist_kpc": np.nan})

    for k in range(n_conatal):
        n = int(rng.integers(12, 41))
        mu = rng.normal(-0.10, 0.15)
        e = rng.uniform(0.05, 0.10, n)                  # realistic GALAH errors
        fe = mu + rng.normal(0, 0.03, n) + rng.normal(0, 1, n) * e
        add_cluster(f"conatal_{k:03d}", fe, e, rng.uniform(0.85, 1.0, n))

    n = 40                                              # the gathered assembly
    e = rng.uniform(0.05, 0.10, n)
    fe = rng.normal(-0.10, 0.22, n) + rng.normal(0, 1, n) * e
    add_cluster("assembly", fe, e, rng.uniform(0.9, 1.0, n))

    # Survey field: quality rows in no cluster (no geometry -> global baseline).
    n_field = 20000
    field_ids = np.arange(5_000_000, 5_000_000 + n_field)
    spec_rows += [{"source_id": int(i), "fe_h": float(f),
                   "e_fe_h": float(e), "r_gal": np.nan, "dist_kpc": np.nan}
                  for i, f, e in zip(field_ids,
                                     rng.normal(-0.1, 0.22, n_field),
                                     rng.uniform(0.05, 0.10, n_field),
                                     strict=True)]
    return pd.DataFrame(mem_rows), pd.DataFrame(spec_rows)


def test_spectro_census_flags_only_the_assembly():
    members, spectro = _synthetic_spectro()
    tab = score_spectro_census(members, spectro, survey="galah")
    assert len(tab) >= 38
    top = tab.iloc[0]
    assert top["cluster"] == "assembly"
    assert bool(top["spectro_candidate"])
    assert top["x_spectro"] >= 2.0 and top["z_census"] >= 4.0
    assert top["field_likeness"] >= 0.5           # mirrors the survey field
    false_pos = tab[tab["spectro_candidate"] & (tab["cluster"] != "assembly")]
    assert len(false_pos) == 0, false_pos["cluster"].tolist()


def test_spectro_low_prob_members_excluded():
    members, spectro = _synthetic_spectro(seed=23, n_conatal=30)
    # Drag every 'assembly' member below the prob threshold: with no
    # quality members left the cluster must vanish from the score table.
    members.loc[members["cluster"] == "assembly", "prob"] = \
        SPECTRO_QUALITY["prob_min"] - 0.05
    tab = score_spectro_census(members, spectro, survey="galah")
    assert "assembly" not in set(tab["cluster"])


def test_spectro_min_members_requirement():
    members, spectro = _synthetic_spectro(seed=29, n_conatal=30)
    rng = np.random.default_rng(2)

    def extra(name, n, base):
        mem = pd.DataFrame({"source_id": np.arange(base, base + n),
                            "cluster": name, "prob": 0.95})
        spec = pd.DataFrame({"source_id": np.arange(base, base + n),
                             "fe_h": rng.normal(-0.1, 0.05, n),
                             "e_fe_h": rng.uniform(0.05, 0.1, n),
                             "r_gal": np.nan, "dist_kpc": np.nan})
        return mem, spec

    m5, s5 = extra("five_members", 5, 8_000_000)      # below the floor
    m6, s6 = extra("six_members", 6, 9_000_000)       # at the floor
    members = pd.concat([members, m5, m6], ignore_index=True)
    spectro = pd.concat([spectro, s5, s6], ignore_index=True)
    tab = score_spectro_census(members, spectro, survey="galah")
    names = set(tab["cluster"])
    assert "five_members" not in names
    assert "six_members" in names


def test_standardize_quality_flag_filter():
    raw = pd.DataFrame({
        "source_id": [1, 2, 3, 4, 5, 6],
        "fe_h": [0.0, 0.1, -0.2, 0.3, np.nan, -0.1],
        "e_fe_h": [0.06, 0.07, 0.08, 0.09, 0.06, 0.9],
        "flag_sp": [0, 1, 0, 0, 0, 0],
        "flag_fe_h": [0, 0, 2, 0, 0, 0],
    })
    out = standardize_spectro(raw)
    # 2: flag_sp != 0; 3: flag_fe_h != 0; 5: NaN fe_h; 6: e_fe_h too big.
    assert set(out["source_id"]) == {1, 4}
    assert out["fe_h"].notna().all()


def test_standardize_without_flag_columns_and_geometry():
    raw = pd.DataFrame({
        "source_id": [10, 11, 12],
        "fe_h": [0.0, -0.3, 0.2],
        "e_fe_h": [0.05, 0.06, 0.07],
        "ra": [266.4, 266.4, 90.0],
        "dec": [-28.9, -28.9, 30.0],
        "plx": [1.0, np.nan, 0.5],
    })
    out = standardize_spectro(raw)
    assert len(out) == 3                          # no flags present -> kept
    assert np.isfinite(out["dist_kpc"]).sum() == 2
    assert np.isfinite(out["r_gal"]).sum() == 2
    # Toward the GC at 1 kpc the Galactocentric radius must shrink below R0.
    row = out[out["source_id"] == 10].iloc[0]
    assert 6.5 < row["r_gal"] < 7.5


def test_resolve_spectro_columns_vizier_names():
    # GALAH-DR3-like VizieR headers, including bracketed [Fe/H] variants.
    cols = ["GaiaEDR3", "RAJ2000", "DEJ2000", "Teff", "__Fe_H_", "e__Fe_H_",
            "flag_sp", "f__Fe_H_", "plx"]
    got = resolve_spectro_columns(cols)
    assert got is not None
    assert got["source_id"] == "GaiaEDR3"
    assert got["fe_h"] == "__Fe_H_"
    assert got["e_fe_h"] == "e__Fe_H_"
    assert got["flag_sp"] == "flag_sp"
    assert got["flag_fe_h"] == "f__Fe_H_"
    assert got["dec"] == "DEJ2000"
    # APOGEE-DR17-like headers: ASPCAPflag serves as the quality flag.
    cols = ["GaiaEDR3", "RAJ2000", "DEJ2000", "[Fe/H]", "e_[Fe/H]",
            "ASPCAPflag"]
    got = resolve_spectro_columns(cols)
    assert got is not None
    assert got["fe_h"] == "[Fe/H]"
    assert got["e_fe_h"] == "e_[Fe/H]"
    assert got["flag_sp"] == "ASPCAPflag"
    # Required columns missing -> clean refusal, not a guess.
    assert resolve_spectro_columns(["RAJ2000", "DEJ2000", "Teff"]) is None


def test_spectro_write_outputs(tmp_path):
    from seti.config import load_config
    from seti.herdsman_b.spectro import spectro_write

    members, spectro = _synthetic_spectro(seed=41, n_conatal=25)
    tab = score_spectro_census(members, spectro, survey="galah")
    joined = members.merge(spectro, on="source_id", how="inner")
    joined["survey"] = "galah"
    cfg = load_config()
    cfg.root = tmp_path
    summary = spectro_write(cfg, tab, joined, ["galah"])
    out = tmp_path / "results" / "herdsman_b"
    assert (out / "spectro_scores.csv").exists()
    assert (out / "spectro_candidates.json").exists()
    assert "Spectroscopic crossmatch" in (out / "REPORT.md").read_text()
    assert summary["n_clusters_spectro"] == len(tab)
    assert "n_spectro_candidates" in summary
    # Re-running replaces (not duplicates) the REPORT section.
    spectro_write(cfg, tab, joined, ["galah"])
    text = (out / "REPORT.md").read_text()
    assert text.count("## Spectroscopic crossmatch (v2)") == 1

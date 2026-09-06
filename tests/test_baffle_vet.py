"""Offline tests for the BAFFLE ``vet`` stage (``seti.baffle.vet``).

No network: every fetcher is a stub over one synthetic sky.  Covered:

* one synthetic frame per veto path and per verdict (BLEND,
  DEBLENDED_COMPONENT, ALLWISE_PHOTOMETRY_WRONG via saturated pixels and via a
  photospheric CatWISE/unWISE residual, W3_INCONSISTENT, SURVIVES_VET,
  INCONCLUSIVE);
* the locus reload from ``locus.json`` and residuals of independent photometry
  against it (including the unWISE nanomaggy -> Vega fallback);
* the upload-query builders composing valid ADQL from quoted TAP_SCHEMA names
  (no ``""`` on the wire; required holes raise);
* the missing-track direct match: the three counters, the control group, the
  truly-missing list and its verdict;
* the stage writing every file with ``NO_DATA_REACHED`` when every fetcher
  raises, and the run.py wiring (``--stage vet``, patch reading
  ``vetted_candidates.csv``, assess carrying the post-vet verdict).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.baffle import locus as L
from seti.baffle import run as R
from seti.baffle import vet as V
from test_baffle import _inject, make_missing, make_population

CFG = V._cfg(None)


# ---------------------------------------------------------------------------
# synthetic sky served through the injectable fetchers
# ---------------------------------------------------------------------------
class Sky:
    """Truth for a handful of candidates, served as Gaia / AllWISE / CatWISE / unWISE."""

    def __init__(self, cands: pd.DataFrame):
        self.cands = cands.reset_index(drop=True)
        self.gaia_extra: list[dict] = []        # extra Gaia sources near candidates
        self.allwise: dict[int, dict] = {}      # per source_id overrides / presence
        self.catwise: dict[int, dict] = {}
        self.unwise: dict[int, dict] = {}
        self.raise_on: set[str] = set()
        self.calls: list[str] = []

    # --- Gaia upload join ---
    def gaia(self, positions, radius_arcsec, label):
        self.calls.append(label)
        if "gaia" in self.raise_on:
            raise RuntimeError("HTTP 500 gaia")
        rows = []
        for _, p in positions.iterrows():
            c = self.cands[self.cands["source_id"] == p["source_id"]].iloc[0]
            rows.append({"target_source_id": int(p["source_id"]), "source_id": int(c["source_id"]),
                         "ra": c["ra"], "dec": c["dec"], "phot_g_mean_mag": c["phot_g_mean_mag"]})
            for e in self.gaia_extra:
                if e["target"] == int(p["source_id"]):
                    rows.append({"target_source_id": int(p["source_id"]), "source_id": e["sid"],
                                 "ra": c["ra"] + e["dra_arcsec"] / 3600.0 / np.cos(np.radians(c["dec"])),
                                 "dec": c["dec"] + e["ddec_arcsec"] / 3600.0,
                                 "phot_g_mean_mag": e["g"]})
        return pd.DataFrame(rows)

    def _matcher(self, name):
        store = getattr(self, name)

        def fetch(positions, radius_arcsec, label):
            self.calls.append(label)
            if name in self.raise_on:
                raise RuntimeError(f"HTTP 500 {name}")
            rows = []
            for _, p in positions.iterrows():
                sid = int(p["source_id"])
                spec = store.get(sid)
                if spec is None or spec.get("absent"):
                    continue
                off = spec.get("offset_arcsec", 0.3)
                row = {"source_id": sid, "ra": p["ra"] + off / 3600.0 / np.cos(np.radians(p["dec"])),
                       "dec": p["dec"], "designation": f"{name}-{sid}"}
                row.update({k: v for k, v in spec.items() if k not in ("absent", "offset_arcsec")})
                rows.append(row)
            return pd.DataFrame(rows)
        return fetch

    def matchers(self):
        return {n: self._matcher(n) for n in ("allwise", "catwise", "unwise")}


def _true_w(cands: pd.DataFrame, locus: L.Locus, band: str) -> np.ndarray:
    """Photospheric W_b for each candidate from the locus."""
    jk = (cands["j_m"] - cands["ks_m"]).to_numpy(float)
    med, _ = locus.predict(jk, cands["lum_class"].astype(str).to_numpy(), band)
    return cands["ks_m"].to_numpy(float) - med


@pytest.fixture(scope="module")
def population():
    return make_population(n=12000, seed=21)


@pytest.fixture(scope="module")
def locus(population):
    return L.fit_locus(population, R.DEFAULTS["locus"])


@pytest.fixture
def cands(population, locus):
    """Six injected two-band-deficit candidates, screened, with lum_class + residuals."""
    df = population.copy()
    idxs = np.argsort((df["ks_m"] - 10.0).abs().to_numpy())[:6]
    for i in idxs:
        df = _inject(df, int(i), 0.8, 0.8)
    r = L.residuals(df, locus, R.DEFAULTS["locus"])
    c = r.iloc[idxs].reset_index(drop=True).copy()
    c["etz"], c["nearby"], c["distance_pc"] = False, False, 1000.0 / c["parallax"]
    return c


def _sky_all_confirm(cands, locus) -> Sky:
    """Every service answers; CatWISE/unWISE confirm the 0.8-mag deficit; W3 normal."""
    sky = Sky(cands)
    w1t, w2t, w3t = (_true_w(cands, locus, b) for b in ("w1", "w2", "w3"))
    for k, sid in enumerate(cands["source_id"]):
        sid = int(sid)
        sky.allwise[sid] = {"w1": w1t[k] + 0.8, "e_w1": 0.03, "w2": w2t[k] + 0.8, "e_w2": 0.03,
                            "w3": w3t[k], "e_w3": 0.05, "w4": 8.0, "e_w4": 0.3,
                            "cc_flags": "0000", "ph_qual": "AAAB", "ext_flag": 0,
                            "nb": 1, "na": 0, "w1sat": 0.0, "w2sat": 0.0,
                            "w1flux": 100.0, "w2flux": 90.0, "w1rchi2": 1.0, "w2rchi2": 1.0}
        sky.catwise[sid] = {"w1": w1t[k] + 0.8, "e_w1": 0.02, "w2": w2t[k] + 0.8, "e_w2": 0.02,
                            "cc_flags": "00", "ab_flags": "00"}
        sky.unwise[sid] = {"w1": w1t[k] + 0.8, "e_w1": 0.02, "w2": w2t[k] + 0.8, "e_w2": 0.02}
    return sky


def _run(cands, locus, sky, cfg=None):
    ledger = V.VetLedger()
    t = V.vet_deficit_candidates(cands, cfg or CFG, locus, ledger, gaia_fetcher=sky.gaia,
                                 matchers=sky.matchers(), locus_cfg=R.DEFAULTS["locus"])
    return t, ledger


# ---------------------------------------------------------------------------
# verdict paths
# ---------------------------------------------------------------------------
def test_confirmed_candidates_survive_the_vet(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    t, ledger = _run(cands, locus, sky)
    assert (t["vet_verdict"] == "SURVIVES_VET").all()
    assert (t["vet_vetoes"] == "").all()
    assert t["vet_notes"].str.contains("catwise_confirms_deficit").all()
    assert (t["independent_class"] == "confirms_deficit").all()
    assert (t["allwise_status"] == "matched").all() and (t["gaia_n_6as"] == 1).all()
    assert (t["w3_status"] == "normal").all()
    assert t["catwise_resid_w1"].between(-0.9, -0.7).all()
    # exactly four upload queries, none per star
    assert len(ledger.entries) == 4 and all(e["status"] == V.QUERY_OK for e in ledger.entries)
    assert len(sky.calls) == 4


def test_a_brighter_gaia_neighbour_within_6as_is_a_blend(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    sid = int(cands.loc[0, "source_id"])
    g = float(cands.loc[0, "phot_g_mean_mag"])
    sky.gaia_extra.append({"target": sid, "sid": 999_001, "dra_arcsec": 3.0, "ddec_arcsec": 2.0, "g": g + 1.0})
    t, _ = _run(cands, locus, sky)
    row = t[t["source_id"] == sid].iloc[0]
    assert row["vet_verdict"] == "BLEND" and "blend_flux_theft" in row["vet_vetoes"]
    assert row["gaia_n_6as"] == 2 and row["gaia_brightest_neighbour_g"] == pytest.approx(g + 1.0)
    assert row["gaia_brightest_neighbour_sep_arcsec"] == pytest.approx(np.hypot(3.0, 2.0), abs=0.05)
    assert (t[t["source_id"] != sid]["vet_verdict"] == "SURVIVES_VET").all()
    # fainter than G + 1.5, or beyond 6": no blend; three sources within 6" is crowded_field
    sky2 = _sky_all_confirm(cands, locus)
    sky2.gaia_extra += [{"target": sid, "sid": 1, "dra_arcsec": 2.0, "ddec_arcsec": 0.0, "g": g + 3.0},
                        {"target": sid, "sid": 2, "dra_arcsec": 0.0, "ddec_arcsec": 4.0, "g": g + 2.0},
                        {"target": sid, "sid": 3, "dra_arcsec": 8.0, "ddec_arcsec": 0.0, "g": g - 5.0}]
    t2, _ = _run(cands, locus, sky2)
    row = t2[t2["source_id"] == sid].iloc[0]
    assert row["vet_verdict"] == "SURVIVES_VET" and "crowded_field" in row["vet_notes"]
    assert row["gaia_n_6as"] == 3 and row["gaia_n_10as"] == 4


def test_deblended_component_and_saturated_pixels(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    s0, s1, s2 = (int(cands.loc[i, "source_id"]) for i in range(3))
    sky.allwise[s0]["nb"] = 2
    sky.allwise[s1]["na"] = 1
    sky.allwise[s2]["w2sat"] = 0.2
    t, _ = _run(cands, locus, sky)
    t = t.set_index("source_id")
    assert t.loc[s0, "vet_verdict"] == "DEBLENDED_COMPONENT"
    assert t.loc[s1, "vet_verdict"] == "DEBLENDED_COMPONENT"
    assert t.loc[s2, "vet_verdict"] == "ALLWISE_PHOTOMETRY_WRONG"
    assert "saturated_pixels" in t.loc[s2, "vet_vetoes"]
    # a blend outranks a deblend flag
    sky.gaia_extra.append({"target": s0, "sid": 5, "dra_arcsec": 1.0, "ddec_arcsec": 0.0,
                           "g": float(cands.loc[0, "phot_g_mean_mag"]) - 1.0})
    t2, _ = _run(cands, locus, sky)
    assert t2.set_index("source_id").loc[s0, "vet_verdict"] == "BLEND"


def test_photospheric_independent_photometry_means_allwise_was_wrong(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    sid = int(cands.loc[0, "source_id"])
    w1t, w2t = _true_w(cands, locus, "w1")[0], _true_w(cands, locus, "w2")[0]
    sky.catwise[sid].update({"w1": w1t, "w2": w2t})
    sky.unwise[sid].update({"w1": w1t, "w2": w2t})
    t, _ = _run(cands, locus, sky)
    row = t.set_index("source_id").loc[sid]
    assert row["vet_verdict"] == "ALLWISE_PHOTOMETRY_WRONG"
    assert "catwise_photospheric" in row["vet_vetoes"] and row["independent_class"] == "photospheric"
    assert abs(row["catwise_resid_w1"]) < 0.1 and abs(row["catwise_sig_w1"]) < 3
    # CatWISE deficit but unWISE photospheric -> the contradiction is resolved as photospheric
    sky.catwise[sid].update({"w1": w1t + 0.8, "w2": w2t + 0.8})
    t2, _ = _run(cands, locus, sky)
    assert t2.set_index("source_id").loc[sid, "vet_verdict"] == "ALLWISE_PHOTOMETRY_WRONG"
    # a mild deficit (-0.15) is neither: INCONCLUSIVE
    sky.catwise[sid].update({"w1": w1t + 0.15, "w2": w2t + 0.15})
    sky.unwise[sid].update({"w1": w1t + 0.15, "w2": w2t + 0.15})
    t3, _ = _run(cands, locus, sky)
    row = t3.set_index("source_id").loc[sid]
    assert row["vet_verdict"] == "INCONCLUSIVE" and row["independent_class"] == "ambiguous"


def test_catwise_and_unwise_missing_is_inconclusive_not_a_survivor(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    sid = int(cands.loc[0, "source_id"])
    sky.catwise[sid] = {"absent": True}
    sky.unwise[sid] = {"absent": True}
    t, _ = _run(cands, locus, sky)
    row = t.set_index("source_id").loc[sid]
    assert row["vet_verdict"] == "INCONCLUSIVE"
    assert row["independent_class"] == "missing" and "catwise_missing" in row["vet_notes"]
    assert row["catwise_status"] == "missing" and row["unwise_status"] == "missing"
    # unWISE alone confirming is enough
    sky.unwise[sid] = dict(_sky_all_confirm(cands, locus).unwise[sid])
    t2, _ = _run(cands, locus, sky)
    assert t2.set_index("source_id").loc[sid, "vet_verdict"] == "SURVIVES_VET"


def test_w3_excess_is_inconsistent_and_w3_deficit_is_noted(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    s0, s1 = int(cands.loc[0, "source_id"]), int(cands.loc[1, "source_id"])
    w3t = _true_w(cands, locus, "w3")
    sky.allwise[s0]["w3"] = w3t[0] - 1.0            # brighter at 12 um: excess
    sky.allwise[s1]["w3"] = w3t[1] + 0.8            # fainter: consistent deficit
    t, _ = _run(cands, locus, sky)
    t = t.set_index("source_id")
    assert t.loc[s0, "vet_verdict"] == "W3_INCONSISTENT" and t.loc[s0, "w3_status"] == "excess"
    assert t.loc[s0, "vet_resid_w3"] > 0.5
    assert t.loc[s1, "vet_verdict"] == "SURVIVES_VET" and "w3_deficit_consistent" in t.loc[s1, "vet_notes"]
    # a W3 with a large error is unmeasured, never a veto
    sky.allwise[s0]["e_w3"] = 0.4
    t2, _ = _run(cands, locus, sky)
    assert t2.set_index("source_id").loc[s0, "w3_status"] == "unmeasured"


def test_w3_status_is_unmeasured_without_a_w3_locus(cands, locus):
    """The first real run's locus.json has no W3 bins: honest, not a veto."""
    no_w3 = L.Locus(bins={c: {b: v for b, v in bands.items() if b != "w3"}
                          for c, bands in locus.bins.items()}, meta=dict(locus.meta))
    sky = _sky_all_confirm(cands, locus)
    t, _ = _run(cands, no_w3, sky)
    assert (t["w3_status"] == "unmeasured").all()
    assert "no W3 locus" in str(t["w3_status_note"].iloc[0])
    assert (t["vet_verdict"] == "SURVIVES_VET").all()


def test_a_failed_service_leaves_the_star_inconclusive_with_the_note(cands, locus):
    sky = _sky_all_confirm(cands, locus)
    sky.raise_on = {"gaia"}
    t, ledger = _run(cands, locus, sky)
    assert (t["vet_verdict"] == "INCONCLUSIVE").all()
    assert t["vet_notes"].str.contains("gaia_neighbours_unavailable").all()
    assert not t["gaia_neighbours_checked"].any()
    assert [e["status"] for e in ledger.entries][0] == V.QUERY_FAILED and ledger.n_failed() == 1
    sky.raise_on = {"catwise", "unwise"}
    t2, _ = _run(cands, locus, sky)
    assert (t2["vet_verdict"] == "INCONCLUSIVE").all()
    assert t2["vet_notes"].str.contains("catwise_unavailable").all()


def test_decide_precedence_table():
    base = {"gaia_neighbours_checked": True, "allwise_status": "matched",
            "independent_class": "confirms_deficit", "w3_status": "normal"}
    assert V.decide(pd.Series(base))[0] == "SURVIVES_VET"
    assert V.decide(pd.Series(dict(base, blend_flux_theft=True, deblended_component=True)))[0] == "BLEND"
    assert V.decide(pd.Series(dict(base, deblended_component=True, w3_status="excess")))[0] == "DEBLENDED_COMPONENT"
    assert V.decide(pd.Series(dict(base, saturated_pixels=True)))[0] == "ALLWISE_PHOTOMETRY_WRONG"
    assert V.decide(pd.Series(dict(base, w3_status="excess")))[0] == "W3_INCONSISTENT"
    assert V.decide(pd.Series(dict(base, allwise_status="missing")))[0] == "INCONCLUSIVE"
    v, vetoes, notes = V.decide(pd.Series(dict(base, crowded_field=True, w3_status="deficit")))
    assert v == "SURVIVES_VET" and vetoes == ""
    assert notes == "crowded_field;catwise_confirms_deficit;w3_deficit_consistent"


# ---------------------------------------------------------------------------
# locus reload, unWISE conversion, geometry
# ---------------------------------------------------------------------------
def test_locus_reload_and_independent_residuals(cands, locus, tmp_path):
    locus.save(tmp_path / "locus.json")
    back = L.Locus.load(tmp_path / "locus.json")
    w1t, w2t = _true_w(cands, locus, "w1"), _true_w(cands, locus, "w2")
    r = V.locus_residuals(cands, w1t + 0.5, np.full(len(cands), 0.02), w2t + 0.5,
                          np.full(len(cands), 0.02), back, R.DEFAULTS["locus"])
    assert np.allclose(r["resid_w1"], -0.5, atol=1e-9) and np.allclose(r["resid_w2"], -0.5, atol=1e-9)
    assert (r["sig_w1"] < -3).all()
    cls = V.classify_independent(r, CFG)
    assert (cls == "confirms_deficit").all()
    r0 = V.locus_residuals(cands, w1t, np.full(len(cands), 0.02), w2t, np.full(len(cands), 0.02), back)
    assert (V.classify_independent(r0, CFG) == "photospheric").all()
    rn = V.locus_residuals(cands, np.full(len(cands), np.nan), None, w2t, None, back)
    assert (V.classify_independent(rn, CFG) == "missing").all()
    # no locus at all: everything NaN -> missing
    rz = V.locus_residuals(cands, w1t, None, w2t, None, None)
    assert np.isnan(rz["resid_w1"]).all()


def test_unwise_nanomaggies_become_vega_magnitudes():
    df = pd.DataFrame({"w1flux": [1000.0, 0.0], "e_w1flux": [10.0, 1.0],
                       "w2flux": [500.0, np.nan], "e_w2flux": [10.0, np.nan]})
    out = V.unwise_vega_mags(df)
    assert out.loc[0, "w1"] == pytest.approx(22.5 - 7.5 - 2.699)
    assert out.loc[0, "w2"] == pytest.approx(22.5 - 2.5 * np.log10(500.0) - 3.339)
    assert out.loc[0, "e_w1"] == pytest.approx(1.0857 * 0.01)
    assert np.isnan(out.loc[1, "w1"]) and np.isnan(out.loc[1, "w2"])
    # a served magnitude wins over the flux conversion
    df2 = pd.DataFrame({"w1": [12.0], "w1flux": [1000.0], "w2flux": [1.0]})
    assert V.unwise_vega_mags(df2).loc[0, "w1"] == 12.0


def test_propagation_and_nearest_per_target():
    # 1"/yr over the 5.5 yr Gaia -> AllWISE baseline moves a star by 5.5"
    ra, dec = V.propagate([10.0], [20.0], [1000.0], [-1000.0], 2016.0, 2010.5)
    assert dec[0] == pytest.approx(20.0 + 5.5 / 3600.0)
    assert ra[0] == pytest.approx(10.0 - 5.5 / 3600.0 / np.cos(np.radians(20.0)))
    assert V.propagate([1.0], [2.0], [np.nan], [np.nan], 2016.0, 2010.5)[0][0] == 1.0
    assert V.separation_arcsec(10.0, 20.0, 10.0, 20.0 + 1 / 3600.0)[()] == pytest.approx(1.0, abs=1e-6)
    targets = pd.DataFrame({"source_id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [0.0, 0.0, 0.0]})
    matches = pd.DataFrame({"source_id": [1, 1, 2, 4],
                            "ra": [10.0 + 2 / 3600, 10.0 + 0.5 / 3600, 20.0 + 5 / 3600, 40.0],
                            "dec": [0.0, 0.0, 0.0, 0.0], "w1": [1.0, 2.0, 3.0, 4.0]})
    near = V.nearest_per_target(matches, targets, 3.0).set_index("source_id")
    assert list(near.index) == [1]
    assert near.loc[1, "w1"] == 2.0 and near.loc[1, "n_within"] == 2
    assert near.loc[1, "sep_arcsec"] == pytest.approx(0.5, abs=1e-3)
    assert V.nearest_per_target(pd.DataFrame(), targets, 3.0).empty


# ---------------------------------------------------------------------------
# query builders
# ---------------------------------------------------------------------------
def test_upload_query_builders_quote_discovered_names_once():
    from seti.baffle import bright

    q = V.gaia_neighbours_upload_query(10.0)
    assert "FROM tap_upload.targets AS u" in q and "JOIN gaiadr3.gaia_source AS g" in q
    assert "CIRCLE('ICRS', u.ra, u.dec, 0.00277778)" in q and "u.source_id AS target_source_id" in q
    served = ['"RAJ2000"', '"DEJ2000"', '"W1mag"', '"e_W1mag"', '"W2mag"', '"ccf"', '"nb"', '"na"',
              '"W1sat"', '"AllWISE"', '"W3mag"']
    res = bright.resolve_aliases(served, V.ALLWISE_ALIASES)
    assert res["ra"] == "RAJ2000" and res["w1sat"] == "W1sat" and "w2sat" not in res
    q2 = V.vizier_upload_query('"II/328/allwise"', res, 3.0)
    assert '""' not in q2 and "TAP_UPLOAD.targets AS u" in q2
    assert "t.W1mag" in q2 and 't."W1mag"' not in q2 and "t.nb" in q2      # plain names bare
    assert "CONTAINS(POINT('ICRS', t.RAJ2000, t.DEJ2000)" in q2
    # a name that needs quoting is quoted exactly once, never doubled
    res2 = bright.resolve_aliases(['"RA-ICRS"', '"DE_ICRS"', '"W1mproPM"'],
                                  dict(V.CATWISE_ALIASES, ra=("RA-ICRS",)))
    q3 = V.vizier_upload_query('"II/365/catwise"', res2, 3.0)
    assert 't."RA-ICRS"' in q3 and '""' not in q3 and "t.DE_ICRS" in q3
    assert "0.00083333" in q2
    with pytest.raises(RuntimeError, match="empty identifier"):
        V.vizier_upload_query('"II/328/allwise"', {"ra": "RAJ2000", "dec": "DEJ2000", "w1": '""'}, 3.0)
    with pytest.raises(RuntimeError, match="position columns not resolved"):
        V.vizier_upload_query('"II/328/allwise"', {"ra": "RAJ2000", "w1": "W1mag"}, 3.0)


def test_upload_matcher_composes_from_discovery_and_canonicalises(monkeypatch):
    from seti.baffle import vet

    seen = {}

    def fake_discover(table, url=None):
        return {"table": table, "names": ["RA_ICRS", "DE_ICRS", "W1mproPM", "e_W1mproPM",
                                          "W2mproPM", "Name", "ccf"], "route": "TAP_SCHEMA"}

    def fake_run(query, *, uploads=None, label="", url=None, retries=3):
        seen["query"] = query
        seen["upload_len"] = len(uploads["targets"])
        return pd.DataFrame({"source_id": [7, 7], "RA_ICRS": [1.0, 1.0001], "DE_ICRS": [2.0, 2.0],
                             "W1mproPM": [10.0, 11.0], "e_W1mproPM": [0.02, 0.03],
                             "W2mproPM": [10.1, 11.1], "Name": ["a", "b"], "ccf": ["00", "00"]})

    monkeypatch.setattr(vet, "discover_columns", fake_discover)
    monkeypatch.setattr(vet, "run_vizier", fake_run)
    m = V.UploadMatcher("catwise", '"II/365/catwise"', V.CATWISE_ALIASES, chunk=1000)
    pos = pd.DataFrame({"source_id": [7], "ra": [1.0], "dec": [2.0]})
    df = m(pos, 3.0, "t")
    assert list(df.columns)[:3] == ["source_id", "designation", "ra"] or "designation" in df.columns
    assert set(df.columns) >= {"source_id", "ra", "dec", "w1", "e_w1", "w2", "designation", "cc_flags"}
    assert "e_w2" not in df.columns                      # not served -> not selected -> absent
    assert '""' not in seen["query"] and "t.W1mproPM" in seen["query"] and seen["upload_len"] == 1
    assert m.routes_used == ["tap_upload"]
    # chunking: two uploads for three rows at chunk=2
    m2 = V.UploadMatcher("catwise", '"II/365/catwise"', V.CATWISE_ALIASES, chunk=2)
    m2(pd.DataFrame({"source_id": [7, 7, 7], "ra": [1.0] * 3, "dec": [2.0] * 3}), 3.0, "t")
    assert m2.routes_used == ["tap_upload", "tap_upload"]
    d = V.default_matchers(CFG)
    assert set(d) == {"allwise", "catwise", "unwise"} and d["unwise"].table == '"II/363/unwise"'


# ---------------------------------------------------------------------------
# missing track
# ---------------------------------------------------------------------------
def _missing_frame():
    m = make_missing(n=300, seed=9)
    m["etz"] = False
    m["nearby"] = False
    m.loc[:19, "nearby"] = True
    m.loc[20:29, "etz"] = True
    return m


def test_missing_track_direct_match_counters_and_truly_missing():
    m = _missing_frame()
    cfg = V._cfg({"missing": {"n_control": 50, "seed": 1}})
    sky = Sky(m.rename(columns={}))
    sids = m["source_id"].astype(int).tolist()
    # 0-14 present within 6", 15-19 at 10", 20-29 (etz) absent; control: all present
    for k, sid in enumerate(sids):
        if k < 15:
            sky.allwise[sid] = {"w1": 8.0, "w2": 8.0, "cc_flags": "0000", "offset_arcsec": 0.5}
        elif k < 20:
            sky.allwise[sid] = {"w1": 5.0, "w2": 5.0, "cc_flags": "0000", "offset_arcsec": 10.0}
        elif k < 30:
            sky.allwise[sid] = {"absent": True}
        else:
            sky.allwise[sid] = {"w1": 9.0, "w2": 9.0, "cc_flags": "0000", "offset_arcsec": 0.4}
    # of the absent ten: 20-24 present in CatWISE (-> absent_in_allwise_only), 25-29 truly missing
    for k in range(20, 25):
        sky.catwise[sids[k]] = {"w1": 8.0, "w2": 8.0, "offset_arcsec": 1.0}

    class NearestSky:
        def __call__(self, positions, radius_arcsec, label):
            sky.calls.append(label)
            if radius_arcsec >= 60:                   # nearest-source query: something at 40"
                return pd.DataFrame({"source_id": positions["source_id"], "ra": positions["ra"],
                                     "dec": positions["dec"] + 40 / 3600.0, "cc_flags": "D000",
                                     "w1": 7.0, "w2": 7.0})
            return sky._matcher("allwise")(positions, radius_arcsec, label)

    matchers = sky.matchers()
    matchers["allwise"] = NearestSky()
    ledger = V.VetLedger()
    out, rep = V.vet_missing(m, cfg, ledger, matchers=matchers)
    assert rep["n_targets"] == 30 + 50 and rep["n_by_group"] == {"control": 50, "nearby": 20, "etz": 10}
    c = rep["counters"]
    assert c["wise_source_present_within_6as"] == 15 + 50
    assert c["wise_source_present_6_to_15as"] == 5
    assert c["no_wise_source_within_15as"] == 10
    assert rep["n_absent_in_allwise"] == 10 and rep["n_truly_missing"] == 5
    assert rep["missing_vet_verdict"] == "TRULY_MISSING_COUNTERPARTS_PENDING (n=5)"
    assert {r["source_id"] for r in rep["truly_missing"]} == set(sids[25:30])
    assert all(r["vet_group"] == "etz" for r in rep["truly_missing"])
    assert rep["control_no_wise_fraction"] == 0.0
    assert rep["counters_by_group"]["etz"]["no_wise_source_within_15as"] == 10
    absent = out[out["wise_status"] == "no_wise_source_within_15as"]
    assert (absent["nearest_allwise_cc_flags"] == "D000").all()
    assert absent["nearest_allwise_sep_arcsec"].between(39, 41).all()
    assert set(out["missing_vet_status"]) >= {"truly_missing", "absent_in_allwise_only",
                                              "wise_source_present_within_6as"}
    # queries: allwise 15", catwise, unwise, allwise nearest -> four uploads, no per-star cones
    assert len(ledger.entries) == 4
    # every counterpart present -> NO_TRULY_MISSING_COUNTERPART
    for k in range(20, 30):
        sky.allwise[sids[k]] = {"w1": 8.0, "w2": 8.0, "cc_flags": "0000", "offset_arcsec": 0.5}
    out2, rep2 = V.vet_missing(m, cfg, V.VetLedger(), matchers=matchers)
    assert rep2["missing_vet_verdict"] == "NO_TRULY_MISSING_COUNTERPART" and rep2["n_absent_in_allwise"] == 0


def test_missing_select_targets_and_failed_service():
    m = _missing_frame()
    t = V.select_missing_targets(m, V._cfg({"missing": {"n_control": 1000, "seed": 3}}))
    assert len(t) == 300 and t["vet_group"].value_counts().to_dict() == {"control": 270, "nearby": 20, "etz": 10}
    t2 = V.select_missing_targets(m, V._cfg({"missing": {"n_control": 0}}))
    assert len(t2) == 30
    sky = Sky(m)
    sky.raise_on = {"allwise", "catwise", "unwise"}
    out, rep = V.vet_missing(m, V._cfg({"missing": {"n_control": 10}}), V.VetLedger(), matchers=sky.matchers())
    assert rep["missing_vet_verdict"] == "NO_DATA_REACHED"
    assert (out["wise_status"] == "unavailable").all()


# ---------------------------------------------------------------------------
# stage + run.py wiring
# ---------------------------------------------------------------------------
def _seed_screen_outputs(out: Path, cands: pd.DataFrame, locus: L.Locus, missing=None):
    out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out / "candidates.csv", index=False)
    d = cands.iloc[:1].copy()
    d["first_veto"], d["vetoes"] = "lpv_colour", "lpv_colour"
    d["source_id"] = 424242
    d.to_csv(out / "deferred_lpv.csv", index=False)
    locus.save(out / "locus.json")
    (missing if missing is not None else make_missing(n=40, seed=2).assign(etz=True, nearby=False)) \
        .to_csv(out / "missing_candidates.csv", index=False)
    (out / "screen.json").write_text(json.dumps({
        "stage": "screen", "verdict": "MIDIR_DEFICIT_CANDIDATES_PENDING_VET (n=6) | "
        "MISSING_COUNTERPART_CANDIDATES_PENDING_VET (n=40)",
        "verdict_deficit": "MIDIR_DEFICIT_CANDIDATES_PENDING_VET (n=6)",
        "verdict_missing": "MISSING_COUNTERPART_CANDIDATES_PENDING_VET (n=40)",
        "deficit": {"n_rows": 100, "funnel": {"n_candidates": 6}},
        "missing": {"n_rows": 40, "funnel": {"n_candidates": 40}}}))


def test_stage_writes_every_file_with_no_data_reached_when_every_fetcher_raises(cands, locus, tmp_path):
    _seed_screen_outputs(tmp_path, cands, locus)
    sky = Sky(cands)
    sky.raise_on = {"gaia", "allwise", "catwise", "unwise"}
    rep = V.run_vet_stage(R.load_baffle_config(None), tmp_path, gaia_fetcher=sky.gaia,
                          matchers=sky.matchers())
    for name in ("vet.json", "vet_table.csv", "vetted_candidates.csv", "missing_vet.json", "missing_vet.csv"):
        assert (tmp_path / name).exists(), name
    assert rep["verdict_deficit_after_vet"] == "NO_DATA_REACHED"
    assert rep["missing_vet_verdict"] == "NO_DATA_REACHED"
    assert rep["n_queries_failed"] == rep["n_queries"] > 0
    assert rep["n_candidates_in"] == 7 and rep["n_from_deferred_lpv"] == 1
    assert rep["verdict_counts"]["INCONCLUSIVE"] == 7 and rep["n_survivors"] == 0
    v = json.loads((tmp_path / "vet.json").read_text())
    assert v["verdict_deficit_after_vet"] == "NO_DATA_REACHED" and len(v["ledger"]) == rep["n_queries"]
    assert (tmp_path / "vetted_candidates.csv").stat().st_size == 0 or \
        len(pd.read_csv(tmp_path / "vetted_candidates.csv")) == 0


def test_stage_end_to_end_through_run_py(cands, locus, tmp_path, monkeypatch):
    _seed_screen_outputs(tmp_path, cands, locus)
    full = pd.concat([cands, pd.read_csv(tmp_path / "deferred_lpv.csv")], ignore_index=True)
    sky = _sky_all_confirm(full, locus)
    sid0 = int(cands.loc[0, "source_id"])
    sky.allwise[sid0]["nb"] = 3                               # one deblended component
    sky.allwise[424242] = {"absent": True}                    # the deferred row: no AllWISE at all
    mm = make_missing(n=40, seed=2).assign(etz=True, nearby=False)
    for sid in mm["source_id"].astype(int):
        sky.allwise[sid] = {"w1": 8.0, "w2": 8.0, "cc_flags": "0000", "offset_arcsec": 0.5}
    rep = R.baffle_run(None, stage="vet", out_root=tmp_path, gaia_fetcher=sky.gaia, matchers=sky.matchers())
    assert rep["verdict_deficit_after_vet"] == "MIDIR_DEFICIT_CANDIDATES_SURVIVE_VET (n=5)"
    assert rep["verdict_counts"] == {"SURVIVES_VET": 5, "BLEND": 0, "DEBLENDED_COMPONENT": 1,
                                     "ALLWISE_PHOTOMETRY_WRONG": 0, "W3_INCONSISTENT": 0, "INCONCLUSIVE": 1}
    assert rep["missing_vet_verdict"] == "NO_TRULY_MISSING_COUNTERPART"
    assert rep["allwise_columns_missing"] == [] and rep["locus_has_w3"] is True
    assert rep["n_queries"] == 4 + 1                          # 4 deficit uploads + 1 missing (no absent group)
    vetted = pd.read_csv(tmp_path / "vetted_candidates.csv")
    assert len(vetted) == 5 and sid0 not in set(vetted["source_id"]) and 424242 not in set(vetted["source_id"])
    table = pd.read_csv(tmp_path / "vet_table.csv")
    assert len(table) == 7 and set(table["vet_source"]) == {"candidates", "deferred_lpv"}
    assert table.set_index("source_id").loc[424242, "vet_notes"].find("allwise_missing") >= 0
    # summary carries the post-vet verdict and drops the query ledger
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["verdict"] == "MIDIR_DEFICIT_CANDIDATES_SURVIVE_VET (n=5) | NO_TRULY_MISSING_COUNTERPART"
    assert s["verdict_screen"].startswith("MIDIR_DEFICIT_CANDIDATES_PENDING_VET")
    assert "ledger" not in s["vet"] and s["missing_vet"]["n_targets"] == 40
    # the patch stage now receives the vetted survivors
    seen = {}
    import sys
    import types
    monkeypatch.setitem(sys.modules, "seti.baffle.patch", types.SimpleNamespace(
        run_patch_stage=lambda c, o, cfg: seen.update(n=len(c)) or {"n_objects": len(c)}))
    prep = R.baffle_run(None, stage="patch", out_root=tmp_path)
    assert seen == {"n": 5} and prep["candidates_source"] == "vetted_candidates.csv"
    # CLI accepts the stage
    a = R.build_parser().parse_args(["--stage", "vet"])
    assert a.stage == "vet" and "vet" in R.STAGES and R.STAGES.index("vet") == R.STAGES.index("patch") - 1


def test_screen_w3_locus_now_fits_without_w3snr(population):
    """The mirror serves no w3snr: the W3 locus must fit on w3mpro_error < 0.2."""
    df = population.copy()
    df["w3snr"] = np.nan
    loc = L.fit_locus(df, R.DEFAULTS["locus"])
    assert loc.has("dwarf", "w3") and loc.has("all", "w3")
    r = L.residuals(df, loc, R.DEFAULTS["locus"])
    assert np.isfinite(r["resid_w3"]).mean() > 0.95
    df2 = df.copy()
    df2["w3mpro_error"] = 0.5
    assert not L.fit_locus(df2, R.DEFAULTS["locus"]).has("all", "w3")
    assert not L.w3_usable(df2, R.DEFAULTS["locus"]).any()


def test_workflow_has_the_vet_step_and_commit_paths():
    import yaml

    text = Path(".github/workflows/baffle.yml").read_text()
    doc = yaml.safe_load(text)
    steps = [s.get("name", "") for s in doc["jobs"]["screen"]["steps"]]
    i_screen = next(i for i, n in enumerate(steps) if n.startswith("Screen"))
    i_vet = next(i for i, n in enumerate(steps) if n.startswith("Vet"))
    i_patch = next(i for i, n in enumerate(steps) if n.startswith("Patch"))
    assert i_screen < i_vet < i_patch
    for f in ("vet.json", "vetted_candidates.csv", "missing_vet.json", "vet_table.csv", "missing_vet.csv"):
        assert f"results/baffle/{f}" in text
    assert "vet-only" in doc["jobs"] and "seti.baffle.run --stage vet" in text
    assert "vet" in doc[True]["workflow_dispatch"]["inputs"]["stage"]["description"]

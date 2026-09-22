"""Offline tests for the SHROUD acquisition rework (2026-09-21).

Short probes, VizieR TAP_SCHEMA discovery, the USNO-B1.0 POSS-I-red-only
reconstruction, catalogue upper limits, the artefact vetoes and the funnel.
No network: every HTTP call is replaced by a canned responder.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse

import numpy as np
import pandas as pd
import pytest
import yaml

from seti.config import load_config
from seti.shroud import acquire as acq
from seti.shroud import classify as cls
from seti.shroud import run as runmod
from seti.shroud import sed as S
from seti.shroud import vet as V

SIGMA_SB = 5.670374419e-8


@pytest.fixture(scope="module")
def sc():
    cfg = load_config()
    with (cfg.root / "config" / "shroud.yaml").open() as fh:
        return yaml.safe_load(fh)


def _enshrouded_row(plate_mag=17.0, t_dust=350.0, ir_scale_factor=1.0,
                    source_id="ENSHROUDED"):
    """Same construction as tests/test_shroud.py: the IR re-radiates the plate flux."""
    teff_grid = [2500, 2800, 3200, 3600, 4000, 4500, 5000, 5800, 6500, 7500, 9000,
                 11000, 15000, 20000]
    bc_min = min(S.bolometric_correction_factor(float(t), "poss1_e") for t in teff_grid)
    f_bol_then = S.mag_to_fnu("poss1_e", plate_mag) * bc_min
    scale = ir_scale_factor * f_bol_then * math.pi / (SIGMA_SB * t_dust ** 4)
    row = {"source_id": source_id, "ra_deg": 190.0, "dec_deg": 42.0,
           "poss1_e": plate_mag, "sample": "usnob1_poss1_red_only",
           "n_ir_neighbours": 1, "ir_local_density_per_deg2": 2000.0,
           # The modern search that found nothing was real and 3.9 mag deeper
           # than the plate detection (Pan-STARRS r = 23.2 vs POSS-I E ~ 20).
           "modern_depth_mag": 23.2, "modern_depth_cats": "ps1,gaia",
           "modern_depth_margin_mag": 23.2 - plate_mag}
    for b in ("w1", "w2", "w3", "w4"):
        row[b] = S.fnu_to_mag(b, scale * S.planck_fnu(t_dust, S.BANDS[b][0]))
        row[f"{b}_err"] = 0.03
    row["2mass_ks_lim"] = 15.3
    row["ps1_r_lim"] = 23.2
    return row


def _plate_defect_row(source_id="DEFECT"):
    return {"source_id": source_id, "ra_deg": 12.0, "dec_deg": 70.0,
            "poss1_e": 19.8, "sample": "usnob1_poss1_red_only",
            "n_ir_neighbours": 0, "ir_local_density_per_deg2": 1500.0,
            "modern_depth_mag": 23.2, "modern_depth_cats": "ps1,gaia",
            "modern_depth_margin_mag": 3.4}


def test_field_grid_is_deterministic_and_inside_the_poss1_footprint(sc):
    g1 = acq.field_grid(30, sc, seed=0)
    g2 = acq.field_grid(30, sc, seed=0)
    assert len(g1) == 30 and g1.equals(g2)
    r = sc["acquire"]["reconstruct"]
    assert (g1["dec_deg"] >= r["dec_min_deg"]).all()
    assert (g1["dec_deg"] <= r["dec_max_deg"]).all()
    assert (g1["glat_deg"].abs() >= r["abs_glat_min_deg"]).all()
    # Growing the request keeps the fields already done (checkpoint reuse).
    g3 = acq.field_grid(60, sc, seed=0)
    assert g3.iloc[:30][["ra_deg", "dec_deg"]].equals(g1[["ra_deg", "dec_deg"]])
    # Well spread: no two centres closer than a field diameter.
    sep = V.angular_separation_arcsec(g1["ra_deg"].to_numpy()[:, None],
                                      g1["dec_deg"].to_numpy()[:, None],
                                      g1["ra_deg"].to_numpy()[None, :],
                                      g1["dec_deg"].to_numpy()[None, :])
    np.fill_diagonal(sep, np.inf)
    assert sep.min() / 3600.0 > 2 * r["field_radius_deg"]


def test_usnob1_field_url_encodes_the_sign_and_the_constraints(sc):
    url = acq.usnob1_field_url(266.4, 65.0, 0.5, sc)
    assert "-c=266.400000%20%2B65.000000" in url, url      # explicit sign, no bare '+'
    c_val = url.split("-c=")[1].split("&")[0]
    assert "+" not in c_val
    assert "-source=I/284/out" in url or "-source=I%2F284%2Fout" in url
    assert "Ndet=1" in url
    assert "R1mag=%3C%3D19.30" in url
    assert "-c.rd=0.5000" in url
    assert "-out.form=TSV" in url
    url_s = acq.usnob1_field_url(10.0, -12.5, 0.5, sc)
    assert "%20-12.500000" in url_s


def test_every_usnob1_query_form_keeps_the_declination_sign_on_the_wire(sc):
    """The IGNITION bug, defended once per rung of the form ladder.

    A literal ``+`` in a query string decodes to a SPACE, so ``-c=266+65``
    reaches VizieR as the unsigned pair ``266 65``, which it cannot read as a
    position: it answers with an empty resource that looks exactly like an
    empty sky.  Every form must percent-encode the sign, and a negative
    declination must survive as a minus.
    """
    for form in acq.USNOB1_QUERY_FORMS:
        for ra, dec, want in ((266.4, 65.0, "%2B65.000000"), (10.0, -12.5, "-12.500000")):
            url = acq.usnob1_field_url(ra, dec, 0.5, sc, form)
            c_val = url.split("-c=")[1].split("&")[0]
            assert "+" not in c_val, (form, url)
            assert c_val.endswith(want), (form, c_val)
            assert "-out.form=TSV" in url, (form, url)
            assert "-c.rd=0.5000" in url, (form, url)


def test_usnob1_query_forms_drop_constraints_down_the_ladder(sc):
    """Each rung asks for strictly less, so a rejected constraint cannot end
    the route: the unconstrained rung is a bare cone and is re-filtered here."""
    seen = []
    for form in acq.USNOB1_QUERY_FORMS:
        u = acq.usnob1_field_url(266.4, 65.0, 0.5, sc, form)
        seen.append(("Ndet=1" in u, "R1mag=" in u))
    assert seen[0] == (True, True)                       # most selective first
    assert seen[-1] == (False, False)                    # bare cone last
    # the last rung asks for every column, so an unknown column name cannot
    # empty the response
    assert "-out.all=" in acq.usnob1_field_url(266.4, 65.0, 0.5, sc, acq.USNOB1_QUERY_FORMS[-1])
    # and the column-list rungs repeat -out=, the spelling VizieR honours
    assert acq.usnob1_field_url(266.4, 65.0, 0.5, sc, "ndet+r1").count("-out=") >= 5


def _asu_body(rows):
    hdr = ["USNO-B1.0", "RAJ2000", "DEJ2000", "Epoch", "pmRA", "pmDE", "Ndet",
           "Flags", "B1mag", "R1mag", "R1S", "R1f", "R1s/g", "B2mag", "R2mag", "Imag"]
    units = ["", "deg", "deg", "yr", "mas/yr", "mas/yr", "", "", "mag", "mag", "", "",
             "", "mag", "mag", "mag"]
    lines = ["#", "#INFO status=OK", "\t".join(hdr), "\t".join(units),
             "\t".join("-" * max(len(h), 1) for h in hdr)]
    for r in rows:
        lines.append("\t".join("" if v is None else str(v) for v in r))
    return "\n".join(lines) + "\n"


def test_normalise_usnob1_keeps_only_poss1_red_only_rows():
    from seti.metronome.acquire import parse_asu_tsv

    body = _asu_body([
        ["1550-0000001", 10.0, 65.0, 1953.7, 0, 0, 1, "", None, 17.2, 1, 100, 11,
         None, None, None],            # E only -> kept
        ["1550-0000002", 10.1, 65.0, 1953.7, 0, 0, 2, "", 18.0, 17.5, 1, 100, 11,
         None, None, None],            # O + E -> not "red only"
        ["1550-0000003", 10.2, 65.0, 1990.1, 0, 0, 1, "s", None, 16.0, 1, 100, 3,
         None, 18.0, None],            # has a POSS-II F detection -> dropped
        ["1550-0000004", 10.3, 65.0, 1953.7, 0, 0, 1, "s", None, 18.9, 1, 100, 2,
         None, None, None],            # E only, spike-flagged -> kept, flag carried
    ])
    raw = parse_asu_tsv(body)
    assert len(raw) == 4
    df = acq.normalise_usnob1_frame(raw, field_id=7)
    assert list(df["usnob_id"]) == ["1550-0000001", "1550-0000004"]
    assert (df["sample"] == "usnob1_poss1_red_only").all()
    assert (df["field_id"] == 7).all()
    assert df["source_id"].iloc[0] == "USNOB-1550-0000001"
    assert df["epoch_poss1"].iloc[0] == pytest.approx(1953.7)
    assert df["poss1_e"].iloc[1] == pytest.approx(18.9)
    assert df["usnob_flags"].iloc[1] == "s"
    assert df["usnob_flags"].iloc[0] == ""
    assert acq.normalise_usnob1_frame(pd.DataFrame()).empty


def test_catalogue_upper_limits_never_become_detections(sc):
    """AllWISE lists a W3/W4 upper limit with a null error and qph 'U'."""
    pos = pd.DataFrame([{"source_id": "S1", "ra_deg": 10.0, "dec_deg": 5.0}])
    allwise = pd.DataFrame([{"source_id": "S1", "angDist": 0.5, "AllWISE": "J0",
                             "W1mag": 15.0, "e_W1mag": 0.03, "W2mag": 14.8,
                             "e_W2mag": 0.05, "W3mag": 12.1, "e_W3mag": np.nan,
                             "W4mag": 8.9, "e_W4mag": np.nan, "qph": "AAUU",
                             "ccf": "0000", "ex": 0}])
    twomass = pd.DataFrame([{"source_id": "S1", "angDist": 0.7, "Jmag": 16.9,
                             "e_Jmag": np.nan, "Hmag": 15.2, "e_Hmag": 0.1,
                             "Kmag": 14.9, "e_Kmag": 0.1, "Qflg": "UAA"}])
    out = acq.join_xmatch_photometry(pos, {"allwise": allwise, "twomass": twomass}, sc)
    r = out.iloc[0]
    assert r["w1"] == 15.0 and r["w2"] == 14.8
    assert np.isnan(r["w3"]) and np.isnan(r["w4"])
    assert r["w3_lim"] == 12.1 and r["w4_lim"] == 8.9
    assert np.isnan(r["2mass_j"]) and r["2mass_j_lim"] == 16.9
    assert r["2mass_h"] == 15.2 and r["2mass_ks"] == 14.9
    sed = V.build_sed(r)
    assert sorted(sed.detected(S.IR_BANDS)) == ["2mass_h", "2mass_ks", "w1", "w2"]
    assert set(sed.limits) >= {"w3", "w4", "2mass_j"}
    assert cls.n_ir_bands(r) == 4


def test_quality_string_alone_marks_a_limit(sc):
    """A table with qph but no error columns still yields limits, not detections."""
    pos = pd.DataFrame([{"source_id": "S1", "ra_deg": 10.0, "dec_deg": 5.0}])
    allwise = pd.DataFrame([{"source_id": "S1", "angDist": 0.5, "W1mag": 15.0,
                             "W2mag": 14.8, "W3mag": 12.1, "W4mag": 8.9, "qph": "ABUU"}])
    out = acq.join_xmatch_photometry(pos, {"allwise": allwise}, sc)
    r = out.iloc[0]
    assert r["w1"] == 15.0 and r["w2"] == 14.8 and np.isnan(r["w3"]) and np.isnan(r["w4"])
    assert r["w3_lim"] == 12.1 and r["w4_lim"] == 8.9


def test_wide_gaia_pull_is_not_a_modern_counterpart(sc):
    """A Gaia star 40" away is a neighbour (and a halo veto if bright), never a match."""
    pos = pd.DataFrame([{"source_id": "S1", "ra_deg": 10.0, "dec_deg": 5.0,
                         "poss1_e": 16.0, "w1": 12.0, "w2": 11.5, "w3": 9.0, "w4": 7.0}])
    gaia = pd.DataFrame([
        {"source_id": "S1", "angDist": 40.0, "Gmag": 8.5, "pmRA": 1.0, "pmDE": 1.0,
         "RA_ICRS": 10.011, "DE_ICRS": 5.0, "Source": 1},
        {"source_id": "S1", "angDist": 55.0, "Gmag": 17.0, "pmRA": 0.0, "pmDE": 0.0,
         "RA_ICRS": 10.015, "DE_ICRS": 5.0, "Source": 2},
    ])
    out = acq.join_xmatch_photometry(pos, {"gaia": gaia}, sc, phot_radius={"gaia": 5.0})
    r = out.iloc[0]
    assert "gaia_g" not in out.columns or np.isnan(r["gaia_g"])
    assert not cls.has_modern_optical(r)
    assert r["bright_nb_gmag"] == 8.5 and r["bright_nb_sep_arcsec"] == 40.0
    flags = V.ledger_vetoes(r, sc)
    assert "BRIGHT_STAR_HALO" in flags
    # Inside 5" the same star IS the counterpart.
    gaia2 = gaia.copy()
    gaia2.loc[0, "angDist"] = 1.0
    out2 = acq.join_xmatch_photometry(pos, {"gaia": gaia2}, sc, phot_radius={"gaia": 5.0})
    assert out2.iloc[0]["gaia_g"] == 8.5 and cls.has_modern_optical(out2.iloc[0])


def test_artefact_flags_are_vetoes(sc):
    base = _enshrouded_row(source_id="FLAGGED")
    assert V.ledger_vetoes(base, sc) == []
    assert "USNOB_SPIKE_FLAG" in V.ledger_vetoes({**base, "usnob_flags": "s"}, sc)
    assert "USNOB_SPIKE_FLAG" not in V.ledger_vetoes({**base, "usnob_flags": "M"}, sc)
    assert "WISE_ARTIFACT_FLAG" in V.ledger_vetoes({**base, "wise_ccf": "d000"}, sc)
    assert "WISE_ARTIFACT_FLAG" in V.ledger_vetoes({**base, "wise_ccf": "0H00"}, sc)
    # Flags on W3/W4 only do not condemn the W1/W2 detection.
    assert "WISE_ARTIFACT_FLAG" not in V.ledger_vetoes({**base, "wise_ccf": "00DD"}, sc)
    assert "WISE_ARTIFACT_FLAG" not in V.ledger_vetoes({**base, "wise_ccf": 0}, sc)
    assert "BRIGHT_STAR_HALO" in V.ledger_vetoes({**base, "bright_nb_gmag": 9.0}, sc)
    assert "BRIGHT_STAR_HALO" not in V.ledger_vetoes({**base, "bright_nb_gmag": 15.0}, sc)


def test_epoch_propagation_uses_the_plate_epoch_when_given(sc):
    """A single-detection USNO-B1.0 object carries its own E-plate date."""
    ra0, dec0 = 200.0, 10.0
    pmra, pmdec = 300.0, 0.0
    t_plate = 1950.2
    dt = sc["epochs"]["gaia_dr3"] - t_plate
    ra_now, dec_now = V.propagate_position(ra0, dec0, pmra, pmdec, dt)
    gaia = pd.DataFrame([{"source_id": "P", "RA_ICRS": float(ra_now),
                          "DE_ICRS": float(dec_now), "pmRA": pmra, "pmDE": pmdec}])
    df = pd.DataFrame([{"source_id": "P", "ra_deg": ra0, "dec_deg": dec0,
                        "epoch_poss1": t_plate}])
    out = runmod.apply_epoch_propagation(df, gaia, sc)
    assert bool(out.loc[0, "pm_recovered"])
    assert out.loc[0, "pm_back_propagated_sep_arcsec"] < 0.05
    # With the default 1953.0 epoch the back-propagation misses by ~0.84".
    df2 = df.drop(columns=["epoch_poss1"])
    out2 = runmod.apply_epoch_propagation(df2, gaia, sc)
    assert out2.loc[0, "pm_back_propagated_sep_arcsec"] > 0.5


def test_report_carries_the_funnel_and_the_coverage(sc, tmp_path):
    rows = [_enshrouded_row(source_id="A"), _plate_defect_row("B"),
            {**_enshrouded_row(source_id="C"), "gaia_g": 17.0}]
    df = runmod.stage_classify(pd.DataFrame(rows), sc)
    df, budgets, fits = runmod.stage_budget(df, sc)
    df = V.vet_table(df, sc, budgets, fits)
    (tmp_path / "field_ledger.json").write_text(json.dumps({
        "n_fields_requested": 3, "radius_deg": 0.5,
        "area_deg2_per_field": math.pi * 0.25,
        "fields": [{"field_id": 0, "status": "ok", "n_raw": 100, "n_poss1_only": 40},
                   {"field_id": 1, "status": "cached", "n_raw": 50, "n_poss1_only": 20},
                   {"field_id": 2, "status": "unreachable", "n_raw": 0,
                    "n_poss1_only": 0}]}))
    cfg = load_config()
    s = runmod.stage_report(cfg, sc, df, {"verdict": "USNOB1_RECONSTRUCTION",
                                          "per_sample_rows": {"usnob1_poss1_red_only": 3}},
                            tmp_path)
    assert s["degraded"] is False
    f = s["funnel"]
    assert f["1_sample"] == 3
    assert f["2_no_modern_optical_within_5arcsec"] == 2
    assert f["3_with_any_ir_detection"] == 2
    assert f["4_ir_present_and_optically_absent"] == 1
    assert f["5_residual_after_population_cascade"] == 1
    assert f["6_survive_every_veto"] == 1
    assert f["7_energy_conserving_obscuration"] == 1
    assert s["sky_coverage"]["n_fields_ok"] == 2
    assert s["sky_coverage"]["area_deg2"] == pytest.approx(2 * math.pi * 0.25, abs=1e-3)
    assert s["sky_coverage"]["fields_failed"] == [2]
    md = (tmp_path / "REPORT.md").read_text()
    assert "## Funnel" in md and "## Sky coverage" in md
    assert any(r["sample"] for r in s["population_by_sample"])


def test_acquire_end_to_end_with_a_dead_svo_and_a_live_vizier(sc, tmp_path, monkeypatch):
    """The whole ladder, offline: SVO dead, TAP_SCHEMA answers, USNO-B1.0 fields answer."""
    calls = []

    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        calls.append((url, timeout, retries))
        if "inta-csic" in url:
            return None, "URLError: timed out"
        if "TAPVizieR" in url:
            return (b'table_name,description\n"J/AJ/159/8/table2","Vanishing sources '
                    b'(Villarroel+ 2020)"\n'), "HTTP 200"
        if "viz-bin/votable" in url:
            p = load_config().root / "results" / "disaplit2" / "vizier_vasco_2020.xml"
            return (p.read_bytes(), "HTTP 200") if p.exists() else (None, "HTTP 404")
        if "asu-tsv" in url and "-meta.all" in url:
            # I/284/out advertises its own columns; 'muPr' is deliberately NOT
            # among them, so the sweep must stop asking for it.
            cols = ["USNO-B1.0", "RAJ2000", "DEJ2000", "Epoch", "pmRA", "pmDE", "Ndet",
                    "Flags", "B1mag", "R1mag", "R1S", "R1f", "R1s/g", "B2mag", "R2mag",
                    "Imag"]
            return ("\n".join(f"#Column\t{c}\t(mag)\tsome description" for c in cols)
                    ).encode(), "HTTP 200"
        if "asu-tsv" in url and "I/284" in urllib.parse.unquote(url):
            m = re.search(r"-c=([\d.]+)%20(%2B|-)([\d.]+)", url)
            ra = float(m.group(1))
            dec = float(m.group(3)) * (-1.0 if m.group(2) == "-" else 1.0)
            tag = f"{int(ra * 10):05d}-{int(abs(dec) * 10):07d}"
            return _asu_body([
                [tag, ra, dec, 1953.7, 0, 0, 1, "", None, 17.2, 1, 100, 11,
                 None, None, None],
                [tag + "b", ra + 0.01, dec, 1953.7, 0, 0, 1, "", 18.0, 17.5, 1, 100, 11,
                 None, None, None],
            ]).encode(), "HTTP 200"
        return None, "HTTP 404"

    monkeypatch.setattr(acq, "http_get", fake_get)
    df, prov = acq.acquire_sample(sc, tmp_path, allow_network=True, n_fields=3)
    assert prov["verdict"] == "USNOB1_RECONSTRUCTION", prov
    assert prov["per_sample_rows"]["usnob1_poss1_red_only"] == 3
    assert (df["sample"] == "usnob1_poss1_red_only").sum() == 3
    assert "epoch_poss1" in df.columns and "poss1_e" in df.columns
    # SVO probes were SHORT: never the 300 s bulk timeout.
    svo = [(u, t) for u, t, _ in calls if "inta-csic" in u]
    assert svo and all(t <= sc["acquire"]["probe_timeout_s"] for _, t in svo)
    assert (tmp_path / "field_ledger.json").exists()
    led = json.loads((tmp_path / "field_ledger.json").read_text())
    assert led["n_fields_ok"] == 3
    # the catalogue's own column list is READ and recorded, but it does not
    # edit the request: a thin -meta.all body must not be able to strip the
    # photometry the selection is defined on (run 35738062833)
    assert led["meta_probe"]["columns"], led["meta_probe"]
    assert led["meta_probe"]["missing"] == ["muPr"], led["meta_probe"]
    assert "muPr" in led["columns_requested"]
    assert (tmp_path / "sample_positions.parquet").exists()
    routes = {r["route"]: r["status"] for r in prov["routes"]}
    assert routes["vizier_tap_schema_discovery"] == "ok"
    assert routes["usnob1_reconstruction"] == "ok"
    # every route is recorded even when it returns nothing: a dead registry and
    # an absent catalogue are findings, not silence
    assert routes["registry_regtap_discovery"] == "unreachable"
    for cat in sc["acquire"]["vizier_direct_catalogues"]:
        assert f"vizier_meta:{cat}" in routes, routes
    assert prov["vizier_tables_discovered"][0]["table_name"] == "J/AJ/159/8/table2"
    # A second call reuses the checkpointed fields instead of refetching them.
    # (The one-off -meta.all probe is not a field fetch and is not counted.)
    def _n_field_fetches():
        return len([u for u, _, _ in calls
                    if "I/284" in urllib.parse.unquote(u) and "-c=" in u])

    n_before = _n_field_fetches()
    acq.acquire_sample(sc, tmp_path, allow_network=True, n_fields=3)
    assert _n_field_fetches() == n_before


def test_acquire_reports_no_data_when_every_route_is_dead(sc, tmp_path, monkeypatch):
    monkeypatch.setattr(acq, "http_get",
                        lambda *a, **k: (None, "URLError: connection refused"))
    df, prov = acq.acquire_sample(sc, tmp_path, allow_network=True, n_fields=2)
    assert len(df) == 0 and prov["verdict"] == "NO_DATA_REACHED"
    assert "nothing was fabricated" in prov["note"]
    assert not (tmp_path / "sample_positions.parquet").exists()


def test_solano_candidates_exclude_the_villarroel_table():
    tabs = pd.DataFrame({"table_name": ["J/AJ/159/8/table2", "J/MNRAS/515/1380/vanish"],
                         "description": ["Vanishing sources (Villarroel+ 2020)",
                                         "POSS I vanishing sources (Solano+ 2022)"]})
    assert acq._solano_candidates(tabs) == ["J/MNRAS/515/1380/vanish"]
    assert acq.unquote_table('"J/AJ/159/8"') == "J/AJ/159/8"


# --- route 0: the IVOA registry ---------------------------------------------
_REGTAP_CSV = (
    b"ivoid,short_name,res_title,access_url,standard_id\n"
    b"ivo://cab.inta-csic.es/vanish-possi,vanish-possi,POSS I vanishing sources,"
    b"http://svocats.cab.inta-csic.es/vanish-possi/cs.php?,"
    b"ivo://ivoa.net/std/ConeSearch\n"
    b"ivo://cab.inta-csic.es/vanish-neowise,vanish-neowise,VASCO NEOWISE counterparts,"
    b"http://example.org/vanish-neowise/cs.php?,ivo://ivoa.net/std/ConeSearch\n"
    b"ivo://cab.inta-csic.es/vasco-web,vasco,VASCO project page,"
    b"http://example.org/vasco/,ivo://ivoa.net/std/TAP\n"
)


def test_root_of_access_url_strips_the_script_and_query():
    f = acq._root_of_access_url
    assert f("http://svocats.cab.inta-csic.es/vanish-possi/cs.php?") == \
        "http://svocats.cab.inta-csic.es/vanish-possi"
    assert f("http://h/x/conesearch?RA=1&DEC=2") == "http://h/x"
    assert f("http://h/vanish-neowise/") == "http://h/vanish-neowise"
    assert f("") == ""


def test_registry_discovery_returns_cone_services_first(sc, monkeypatch):
    seen = []

    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        seen.append(url)
        if "reg.g-vo.org" in url:
            return _REGTAP_CSV, "HTTP 200"
        return None, "URLError: timed out"

    monkeypatch.setattr(acq, "http_get", fake_get)
    roots, prov = acq.discover_registry_services(sc)
    assert prov.status == "ok"
    # the two ConeSearch interfaces come before the TAP one
    assert roots[:2] == ["http://svocats.cab.inta-csic.es/vanish-possi",
                         "http://example.org/vanish-neowise"]
    assert "http://example.org/vasco" in roots
    # the first mirror that answered was enough: no second mirror was queried
    assert sum(1 for u in seen if "g-vo.org" in u or "euro-vo" in u) == 1
    # the ADQL asked the registry, not a guessed path
    assert "rr.resource" in urllib.parse.unquote(seen[0])


def test_registry_discovery_is_honest_when_every_mirror_is_dead(sc, monkeypatch):
    monkeypatch.setattr(acq, "http_get",
                        lambda *a, **k: (None, "URLError: connection refused"))
    roots, prov = acq.discover_registry_services(sc)
    assert roots == []
    assert prov.status == "unreachable"
    assert len(prov.attempts) == len(sc["acquire"]["regtap_endpoints"])
    assert all(not a["ok"] for a in prov.attempts)


# --- looking a VizieR catalogue up BY NAME ----------------------------------
def test_vizier_catalogue_meta_separates_absent_from_present(sc, monkeypatch):
    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        if "J%2FMNRAS%2F515%2F1380" in url or "J/MNRAS/515/1380" in url:
            return (b"#RESOURCE=J/MNRAS/515/1380\n"
                    b"#Table\tJ/MNRAS/515/1380/vanish\tthe by-product\n"
                    b"#Table\tJ/MNRAS/515/1380/table1\tsummary\n"), "HTTP 200"
        return b"#***Nothing found in the metadata\n", "HTTP 200"

    monkeypatch.setattr(acq, "http_get", fake_get)
    tabs, prov = acq.vizier_catalogue_meta("J/MNRAS/515/1380", sc)
    assert prov.status == "ok"
    assert tabs == ["J/MNRAS/515/1380/table1", "J/MNRAS/515/1380/vanish"]
    tabs2, prov2 = acq.vizier_catalogue_meta("J/NOPE/1/1", sc)
    assert tabs2 == [] and prov2.status == "asu_error"
    assert any("Nothing found" in n for n in prov2.notes)


def test_vizier_catalogue_meta_records_an_unreachable_service(sc, monkeypatch):
    monkeypatch.setattr(acq, "http_get", lambda *a, **k: (None, "HTTP 503"))
    tabs, prov = acq.vizier_catalogue_meta("J/MNRAS/515/1380", sc)
    assert tabs == [] and prov.status == "unreachable"
    assert prov.attempts[0]["detail"] == "HTTP 503"


# --- run 35738062833: rows that cannot express the selection ----------------
def _positions_only_body(rows):
    """What VizieR returned when only positional columns were requested."""
    hdr = ["USNO-B1.0", "RAJ2000", "DEJ2000", "Epoch", "pmRA", "pmDE"]
    lines = ["#", "#INFO status=OK", "\t".join(hdr),
             "\t".join(["", "deg", "deg", "yr", "mas/yr", "mas/yr"]),
             "\t".join("-" * max(len(h), 1) for h in hdr)]
    for r in rows:
        lines.append("\t".join(str(v) for v in r))
    return "\n".join(lines) + "\n"


def test_a_field_without_the_plate_magnitudes_is_rejected_not_counted_as_zero(sc):
    """The exact shape of run 35738062833: 1380-5899 rows per field, zero
    reconstructed sources, because the answer carried no photometry.

    'POSS-I red and nothing else' is a statement about which plate magnitudes
    are PRESENT, so a frame without them yields a guaranteed zero that says
    nothing about the sky.  Such a rung must be rejected, not believed.
    """
    calls = []

    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        calls.append(url)
        # every column-list rung answers with positions only; -out.all is full
        if "-out.all=" in url:
            return _asu_body([
                ["1550-0000001", 10.0, 65.0, 1953.7, 0, 0, 1, "", None, 17.2, 1, 100,
                 11, None, None, None],
                ["1550-0000002", 10.1, 65.0, 1953.7, 0, 0, 2, "", 18.0, 17.5, 1, 100,
                 11, None, None, None],
            ]).encode(), "HTTP 200"
        return _positions_only_body([
            ["1550-0000001", 10.0, 65.0, 1953.7, 0, 0],
            ["1550-0000002", 10.1, 65.0, 1953.7, 0, 0],
        ]).encode(), "HTTP 200"

    import seti.shroud.acquire as m
    orig, m.http_get = m.http_get, fake_get
    try:
        raw, form, attempts = m.fetch_usnob1_field(10.0, 65.0, 0.5, sc)
    finally:
        m.http_get = orig

    # the positions-only rungs were NOT accepted even though they had rows
    rejected = [a for a in attempts if a["missing_required"]]
    assert rejected, attempts
    assert all(a["n_raw"] == 2 for a in rejected)        # rows, but unusable rows
    assert all("R1mag" in a["missing_required"] for a in rejected)
    # the ladder fell through to the rung that names no columns
    assert form == "none_allcols", [a["form"] for a in attempts]
    assert "R1mag" in raw.columns and len(raw) == 2
    # and that frame CAN express the selection: one POSS-I-red-only object
    out = m.normalise_usnob1_frame(raw, 0, r1_max=19.3, ndet=1)
    assert len(out) == 1 and out["source_id"].iloc[0] == "USNOB-1550-0000001"


def test_a_field_with_no_usable_rung_returns_an_empty_frame_not_a_bad_one(sc):
    """If NO rung can express the selection, the field is empty --- it must not
    silently hand back the last (unusable) answer as if it were the sample."""
    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        return _positions_only_body([["1550-0000001", 10.0, 65.0, 1953.7, 0, 0]]).encode(), \
            "HTTP 200"

    import seti.shroud.acquire as m
    orig, m.http_get = m.http_get, fake_get
    try:
        raw, form, attempts = m.fetch_usnob1_field(10.0, 65.0, 0.5, sc)
    finally:
        m.http_get = orig
    assert len(raw) == 0 and form == ""
    assert len(attempts) == len(acq.USNOB1_QUERY_FORMS)
    assert all(a["n_raw"] == 1 and a["missing_required"] for a in attempts)


# --- the depth kill: an absence is only as good as the search behind it -----
def test_a_failed_modern_xmatch_is_not_a_disappearance(sc):
    """The dominant way this channel could fabricate a detection.

    If the Pan-STARRS and Gaia X-Matches simply error, every source in the run
    comes out with an empty ps1_r / gaia_g -- and empty reads as *gone*.  The
    depth record must make that impossible: a catalogue that did not answer
    establishes nothing.
    """
    pos = pd.DataFrame({"source_id": ["A", "B"], "ra_deg": [10.0, 20.0],
                        "dec_deg": [40.0, -50.0]})
    dead = {"ps1": {"status": "unreachable"}, "gaia": {"status": "unreachable"}}
    depth, cats = acq.modern_optical_depth(pos, dead, sc)
    assert depth.isna().all(), depth
    assert (cats == "").all()
    # ... and every such source trips the veto rather than surviving
    for sid, d in zip(pos["source_id"], depth, strict=False):
        flags = V.ledger_vetoes({"source_id": sid, "poss1_e": 18.0,
                                 "modern_depth_mag": d,
                                 "modern_depth_margin_mag": float("nan")}, sc)
        assert "MODERN_OPTICAL_NOT_SEARCHED" in flags, flags


def test_depth_follows_the_footprint_not_just_the_query_status(sc):
    """Pan-STARRS stops at dec = -30, so a southern source has only Gaia --
    three magnitudes shallower -- behind its absence."""
    pos = pd.DataFrame({"source_id": ["north", "south"], "ra_deg": [10.0, 20.0],
                        "dec_deg": [40.0, -50.0]})
    live = {"ps1": {"status": "ok"}, "gaia": {"status": "ok"}}
    depth, cats = acq.modern_optical_depth(pos, live, sc)
    lim = sc["modern_optical"]["limits"]
    assert depth.iloc[0] == lim["ps1"]["mag"]        # PS1 is the deeper one
    assert depth.iloc[1] == lim["gaia"]["mag"]       # outside the PS1 footprint
    assert "ps1" in cats.iloc[0] and "ps1" not in cats.iloc[1]


def test_a_shallow_absence_is_vetoed_and_a_deep_one_is_not(sc):
    """POSS-I E ~ 20.  Absent from Gaia (G = 20.7) alone means it faded by
    0.7 mag -- ordinary variability.  Absent from Pan-STARRS means >= 3 mag."""
    base = {"source_id": "S", "ra_deg": 10.0, "dec_deg": 40.0, "poss1_e": 20.0}
    shallow = V.ledger_vetoes({**base, "modern_depth_mag": 20.7,
                               "modern_depth_margin_mag": 0.7}, sc)
    assert "MODERN_DEPTH_INSUFFICIENT" in shallow, shallow
    deep = V.ledger_vetoes({**base, "modern_depth_mag": 23.2,
                            "modern_depth_margin_mag": 3.2}, sc)
    assert "MODERN_DEPTH_INSUFFICIENT" not in deep, deep
    assert "MODERN_OPTICAL_NOT_SEARCHED" not in deep, deep


def test_a_live_catalogue_that_is_only_partly_deep_is_recorded_per_source(sc):
    """One catalogue up, one down: the depth is whatever ACTUALLY answered."""
    pos = pd.DataFrame({"source_id": ["a"], "ra_deg": [10.0], "dec_deg": [40.0]})
    depth, cats = acq.modern_optical_depth(
        pos, {"ps1": {"status": "unreachable"}, "gaia": {"status": "cached"}}, sc)
    assert depth.iloc[0] == sc["modern_optical"]["limits"]["gaia"]["mag"]
    assert cats.iloc[0] == "gaia"


def test_a_run_whose_photometry_never_happened_says_so(sc, tmp_path):
    """The exact failure of run 35738062833's analyze job.

    Its photometry job was cancelled, so analyze reported
    ``3_with_any_ir_detection: 0`` for 127 sources that were never searched.
    A zero from a search that did not happen must never read like a zero from
    a search that found nothing.
    """
    rows = [{"source_id": f"S{i}", "ra_deg": 10.0 + i, "dec_deg": 40.0,
             "poss1_e": 18.5, "sample": "vasco2020_surviving_candidates"}
            for i in range(5)]
    df = runmod.stage_classify(pd.DataFrame(rows), sc)
    df, budgets, fits = runmod.stage_budget(df, sc)
    df = V.vet_table(df, sc, budgets, fits)
    s = runmod.stage_report(load_config(), sc, df,
                            {"verdict": "VIZIER_FALLBACK"}, tmp_path)
    assert s["degraded"] is True
    assert s["funnel"]["3_with_any_ir_detection"] == 0
    why = " ".join(s["degraded_reason"])
    assert "NO_INFRARED_SEARCH" in why, s["degraded_reason"]
    assert "NO_MODERN_OPTICAL_SEARCH" in why, s["degraded_reason"]
    assert s["photometry_reached"]["infrared_searched"] is False
    assert s["photometry_reached"]["modern_optical_searched"] is False
    # And the human-readable report leads with it, not with the zero.
    rep = (tmp_path / "REPORT.md").read_text()
    assert "DEGRADED" in rep and "not about the sky" in rep


def test_photometry_that_did_happen_is_not_called_degraded_for_it(sc, tmp_path):
    """The other half: real photometry must not trip the new reason."""
    rows = [_enshrouded_row(source_id="A"), _plate_defect_row("B")]
    df = runmod.stage_classify(pd.DataFrame(rows), sc)
    df, budgets, fits = runmod.stage_budget(df, sc)
    df = V.vet_table(df, sc, budgets, fits)
    s = runmod.stage_report(load_config(), sc, df,
                            {"verdict": "USNOB1_RECONSTRUCTION"}, tmp_path)
    assert s["photometry_reached"]["infrared_searched"] is True
    assert s["photometry_reached"]["modern_optical_searched"] is True
    assert s["degraded_reason"] == []
    assert s["degraded"] is False


def test_the_svo_probe_ladder_runs_under_a_clock(sc, monkeypatch):
    """A dead service must not be able to eat the run that would have worked.

    Run 35741075121 spent > 45 min in this one step, because the number of
    roots is contributed by the registry and by an index scrape, not by this
    channel.  The route AFTER it is the one that can restore the sample.
    """
    t = [0.0]

    def slow_probe(url, cfg_, data=None, timeout=None):
        t[0] += 25.0                       # every root times out, as they do
        return None, "URLError: timed out"

    monkeypatch.setattr(acq, "_probe", slow_probe)
    monkeypatch.setattr(acq.time, "time", lambda: t[0])
    roots = [f"http://dead-{i}.example" for i in range(200)]
    root, _url, prov = acq.probe_svo_catalog("vanish_neowise", roots, sc,
                                             budget_s=300.0)
    assert root is None
    d = prov.as_dict()
    assert d["status"] == "budget_exhausted", d
    # It stopped early: 200 roots x 5 forms x 25 s is 7 hours.
    assert len(d["attempts"]) < 40, len(d["attempts"])
    # And it says so, so a reader cannot mistake the clock for the sky.
    assert any("not about the service" in n for n in d["notes"]), d["notes"]


def test_a_live_root_inside_the_budget_is_still_found(sc, monkeypatch):
    """The clock must not cost the channel a service that does answer."""
    def probe(url, cfg_, data=None, timeout=None):
        if "live" in url and "RA=" in url:
            return (b"RA\tDEC\tW1mag\tW2mag\n"
                    b"180.000000\t0.000000\t15.1\t14.8\n"
                    b"180.001000\t0.001000\t16.2\t15.9\n"), "HTTP 200"
        if "live" in url:
            return b"<html><a href='cs.php'>cone</a></html>", "HTTP 200"
        return None, "URLError: timed out"

    monkeypatch.setattr(acq, "_probe", probe)
    root, url, prov = acq.probe_svo_catalog(
        "vanish_neowise", ["http://dead.example", "http://live.example"], sc,
        budget_s=600.0)
    assert root == "http://live.example", prov.as_dict()
    assert "RA=" in url
    assert prov.status == "ok"


# ===========================================================================
# The second-digitisation reachability probe.
# ===========================================================================
def test_second_digitisation_probe_reports_a_live_route(sc, monkeypatch):
    """An independent scan reachable: the route is 'ok' and names what answered."""
    seen = []

    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        seen.append(url)
        if "TAPVizieR" in url and "SuperCOSMOS" in urllib.parse.unquote(url):
            return (b'table_name,description\n'
                    b'"II/341/sss","SuperCOSMOS Sky Survey (Hambly+ 2001)"\n'), "HTTP 200"
        if "ssa.roe.ac.uk" in url:
            return b"table_name\nssa.Source\n", "HTTP 200"
        if "-meta.all" in url:
            return b"#Column\tRAJ2000\t(deg)\tRight ascension\n", "HTTP 200"
        return None, "HTTP 404"

    monkeypatch.setattr(acq, "http_get", fake_get)
    d = acq.probe_second_digitisation(sc).as_dict()
    assert d["route"] == "second_digitisation_probe"
    assert d["status"] == "ok"
    assert d["n_rows"] >= 1
    assert any("II/341/sss" in n for n in d["notes"]), d["notes"]
    assert any("ssa.roe.ac.uk" in u for u in seen)


def test_second_digitisation_probe_is_honest_when_nothing_answers(sc, monkeypatch):
    """Nothing answering is a statement about the archives, not the sky ---
    and it must FORBID the kill rather than silently skip it."""
    cap = max(int(sc["acquire"]["second_digitisation"]["probe_timeout_s"]),
              int(sc["acquire"].get("tap_timeout_s", 120)))

    def dead(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        # A dead host in Edinburgh must not be able to eat the acquisition
        # budget: run 30203741898 lost a whole 70-minute job to a 300 s retry
        # ladder against a dead SVO host.
        assert timeout <= cap, (url, timeout)
        return None, "URLError: timed out"

    monkeypatch.setattr(acq, "http_get", dead)
    d = acq.probe_second_digitisation(sc).as_dict()
    assert d["status"] == "unreachable"
    assert d["n_rows"] == 0
    assert any("no source may be vetoed" in n for n in d["notes"]), d["notes"]
    # Every endpoint tried is on the record, with its error verbatim.
    assert d["attempts"] and all(a["ok"] is False for a in d["attempts"])
    assert any("ssa.roe.ac.uk" in a["url"] for a in d["attempts"])


def test_second_digitisation_probe_runs_inside_the_acquisition(sc, tmp_path,
                                                               monkeypatch):
    """It is part of the route ledger, not something someone must remember."""
    def fake_get(url, timeout=300, retries=4, backoff=8.0, data=None, headers=None):
        if "TAPVizieR" in url and "SuperCOSMOS" in urllib.parse.unquote(url):
            return b'table_name,description\n"II/341/sss","SuperCOSMOS"\n', "HTTP 200"
        return None, "HTTP 404"

    monkeypatch.setattr(acq, "http_get", fake_get)
    _df, prov = acq.acquire_sample(sc, tmp_path, allow_network=True, n_fields=1)
    routes = {r["route"]: r["status"] for r in prov["routes"]}
    assert "second_digitisation_probe" in routes, routes

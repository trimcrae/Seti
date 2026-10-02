"""Synthetic VSX/Gaia association controls, not acquired counterpart evidence."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from seti.parallax4 import deepvet as DV
from seti.parallax4 import vsx_association as VA


def target(**changes):
    return {
        "source_id": "1000000000000000077", "ra": 10.0, "dec": -5.0,
        "ref_epoch": 2016.0, "pmra": 0.0, "pmdec": 0.0, **changes,
    }


def receipt(**changes):
    # Synthetic, caller-reviewed input only. No URL/hash content was acquired
    # and the program never authenticates this trust-boundary assertion.
    return {
        "schema": "vsx-gaia-association-v1", "review_status": "CALLER_REVIEWED",
        "catalogue_table": "B/vsx/vsx", "gaia_release": "DR3",
        "gaia_source_id": "1000000000000000077", "vsx_oid": "9007199254740993",
        "position_frame": "ICRS", "position_epoch_jyear": 2000.0,
        "max_separation_arcsec": 1.0,
        "evidence_url": "https://example.invalid/synthetic-vsx-gaia-receipt",
        "evidence_sha256": "a" * 64, **changes,
    }


def native_row(**changes):
    return {
        "OID": "9007199254740993", "Name": "synthetic-target", "Type": "EA",
        "Period": 1.2, "RAJ2000": 10.0, "DEJ2000": -5.0,
        "Epoch": 2450000.0, **changes,
    }


def synthetic_association(**changes):
    return VA.audit_candidates(pd.DataFrame([native_row(**changes)]),
                               target=target(), receipt=receipt())


def test_original_neighbor_witness_now_remains_incomplete(monkeypatch):
    row = native_row(Name="synthetic-neighbor",
                     RAJ2000=10 + 9 / 3600 / np.cos(np.deg2rad(-5)))
    monkeypatch.setattr(DV, "_tap_rows", lambda *a, **kw: pd.DataFrame([row]))
    result = DV.vsx_type(10, -5)
    fate, flags = DV.classify_fate(result)
    assert result["vsx_candidates"][0]["Type"] == "EA"
    assert result["vsx_candidates"][0]["OID"] == "9007199254740993"
    assert "vsx_type" not in result
    assert fate == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"
    assert "vsx_target_association_unresolved" in flags


def test_caller_reviewed_cross_id_and_consistent_position_positive_control():
    result = synthetic_association()
    assert result["vsx_association_status"] == VA.SUPPORTED
    assert result["vsx_receipt_authenticated_by_program"] is False
    assert result["vsx_position_epoch_status"] == "CALLER_REVIEWED_RECEIPT"
    assert DV.classify_fate(result)[0] == "KNOWN_ECLIPSING_BINARY(VSX:EA)"


@pytest.mark.parametrize(("ra", "dec", "pmra", "pmdec", "epoch"), [
    (359.999, 0.0, 1000.0, 0.0, 2032.0),
    (0.001, 0.0, 1000.0, 0.0, 2000.0),
    (10.0, 60.0, 1000.0, 0.0, 2000.0),
    (10.0, 89.999, 1000.0, 2000.0, 2032.0),
    (240.0, -89.999, -1000.0, -2000.0, 2032.0),
    (10.0, -5.0, 0.0, 0.0, 2016.0),
])
def test_spherical_motion_matches_independent_great_circle(ra, dec, pmra, pmdec, epoch):
    astrometry = target(ra=ra, dec=dec, pmra=pmra, pmdec=pmdec)
    speed = np.hypot(pmra, pmdec) * np.deg2rad(1 / 3600000)
    angle = np.arctan((epoch - 2016) * speed) * u.rad
    position_angle = np.arctan2(pmra, pmdec) * u.rad
    expected = SkyCoord(ra * u.deg, dec * u.deg).directional_offset_by(position_angle, angle)
    vector = VA.propagated_direction(astrometry, epoch)
    actual = SkyCoord(x=vector[0], y=vector[1], z=vector[2], representation_type="cartesian")
    assert actual.separation(expected).arcsec < 1e-6


def test_query_center_moves_to_explicit_position_epoch(monkeypatch):
    queries = []
    astrometry = target(ra=0.0, dec=0.0, pmra=1000.0)
    moved_ra = np.rad2deg(np.arctan(-16 * np.deg2rad(1 / 3600))) % 360
    rows = pd.DataFrame([native_row(RAJ2000=moved_ra, DEJ2000=0.0)])
    def tap(url, query):
        queries.append(query)
        return rows
    monkeypatch.setattr(DV, "_tap_rows", tap)
    result = DV.vsx_type(0, 0, target_astrometry=astrometry, association_receipt=receipt())
    assert result["vsx_query_ra"] == pytest.approx(moved_ra)
    assert result["vsx_query_position_epoch_jyear"] == 2000
    assert f"{moved_ra:.10f}" in queries[0]
    assert result["vsx_association_status"] == VA.SUPPORTED
    assert result["vsx_candidates"][0]["association_separation_arcsec"] < 1e-6


@pytest.mark.parametrize("field", ["ra", "dec", "ref_epoch", "pmra", "pmdec"])
def test_missing_target_astrometry_is_explicit(field):
    astrometry = target()
    astrometry.pop(field)
    result = VA.audit_candidates(pd.DataFrame([native_row()]),
                                 target=astrometry, receipt=receipt())
    assert result["vsx_association_status"] == "GAIA_ASTROMETRY_INCOMPLETE"
    assert "vsx_type" not in result


@pytest.mark.parametrize("astrometry", [
    target(source_id=1000000000000000000.0), target(source_id=True),
    target(source_id=None), target(ra=np.nan), target(dec=91),
    target(pmra=np.inf), target(pmdec=False), target(ref_epoch=np.nan),
])
def test_invalid_or_lossy_gaia_context_refuses(astrometry):
    result = VA.audit_candidates(pd.DataFrame([native_row()]),
                                 target=astrometry, receipt=receipt())
    assert result["vsx_association_status"] != VA.SUPPORTED
    assert "vsx_type" not in result


@pytest.mark.parametrize("changes", [
    {"schema": "other"}, {"review_status": "VERIFIED"},
    {"catalogue_table": "Gaia"}, {"gaia_release": "DR2"}, {"position_frame": "FK5"},
    {"gaia_source_id": "9007199254740993"}, {"vsx_oid": "1000000000000000077"},
    {"gaia_source_id": 1000000000000000000.0}, {"vsx_oid": 9007199254740992.0},
    {"position_epoch_jyear": None}, {"position_epoch_jyear": np.nan},
    {"position_epoch_jyear": True}, {"max_separation_arcsec": 0},
    {"max_separation_arcsec": -1}, {"max_separation_arcsec": np.inf},
    {"evidence_url": "http://example.invalid/receipt"}, {"evidence_url": "https://bad url"},
    {"evidence_sha256": "a" * 63}, {"evidence_sha256": "G" * 64},
])
def test_unsupported_receipt_bindings_refuse(changes):
    result = VA.audit_candidates(pd.DataFrame([native_row()]),
                                 target=target(), receipt=receipt(**changes))
    assert result["vsx_association_status"] != VA.SUPPORTED
    assert "vsx_type" not in result


def test_native_photometric_epoch_never_supplies_position_epoch():
    r = receipt()
    r.pop("position_epoch_jyear")
    result = VA.audit_candidates(pd.DataFrame([native_row(Epoch=2000)]),
                                 target=target(), receipt=r)
    assert result["vsx_association_status"] == "POSITION_EPOCH_UNKNOWN"
    assert "vsx_type" not in result


def test_wrong_native_object_id_cannot_use_a_close_position():
    result = VA.audit_candidates(pd.DataFrame([native_row(OID="9007199254740992")]),
                                 target=target(), receipt=receipt())
    assert result["vsx_association_status"] == "RECEIPT_OBJECT_NOT_POSITION_MATCH"


@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_position_candidates_are_ambiguous_in_either_order(reverse):
    rows = [native_row(), native_row(OID="9007199254740992", Type="RRAB")]
    result = VA.audit_candidates(pd.DataFrame(rows[::-1] if reverse else rows),
                                 target=target(), receipt=receipt())
    assert result["vsx_association_status"] == "AMBIGUOUS_POSITION_CANDIDATES"
    assert len(result["vsx_candidates"]) == 2 and "vsx_type" not in result


def test_duplicate_native_ids_are_not_a_unique_association():
    result = VA.audit_candidates(pd.DataFrame([native_row(), native_row(RAJ2000=10.001)]),
                                 target=target(), receipt=receipt())
    assert result["vsx_association_status"] == "DUPLICATE_NATIVE_OBJECT_IDS"


@pytest.mark.parametrize("changes", [
    {"OID": 9007199254740992.0}, {"OID": None}, {"RAJ2000": np.nan},
    {"DEJ2000": None}, {"RAJ2000": 361}, {"DEJ2000": -91},
])
def test_invalid_candidate_identity_or_position_prevents_uniqueness(changes):
    result = VA.audit_candidates(pd.DataFrame([native_row(), native_row(**changes)]),
                                 target=target(), receipt=receipt())
    assert result["vsx_association_status"] == "CANDIDATE_ID_OR_POSITION_INVALID"


def test_offset_receipt_cannot_promote_neighbor():
    result = synthetic_association(RAJ2000=10 + 9 / 3600 / np.cos(np.deg2rad(-5)))
    assert result["vsx_association_status"] == "POSITION_OFFSET"
    assert result["vsx_candidates"][0]["association_separation_arcsec"] == pytest.approx(9.0)
    assert "vsx_type" not in result


@pytest.mark.parametrize("n_rows", [100, 101])
def test_bounded_response_boundary_retains_rows_and_refuses_truncation(n_rows):
    rows = [native_row()] + [
        native_row(OID=str(1000 + k), RAJ2000=10 + 9 / 3600 / np.cos(np.deg2rad(-5)))
        for k in range(n_rows - 1)
    ]
    result = VA.audit_candidates(pd.DataFrame(rows), target=target(), receipt=receipt())
    expected = VA.SUPPORTED if n_rows == 100 else "CONE_RESPONSE_TRUNCATED"
    assert result["vsx_association_status"] == expected
    assert result["vsx_n_rows_returned"] == result["vsx_n_rows_preserved"] == n_rows
    assert [x["OID"] for x in result["vsx_candidates"]] == [x["OID"] for x in rows]


def test_zero_rows_at_unknown_epoch_are_not_catalogue_absence(monkeypatch):
    monkeypatch.setattr(DV, "_tap_rows", lambda *a, **kw: pd.DataFrame())
    result = DV.vsx_type(10, -5)
    assert result["vsx_status"] == "NO_ROWS_AT_QUERY_POSITION"
    assert result["vsx_query_position_epoch_status"] == "UNVERIFIED"
    assert DV.classify_fate(result)[0] == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"


def test_query_error_remains_availability_gap(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic service refusal")
    monkeypatch.setattr(DV, "_tap_rows", failed)
    result = DV.vsx_type(10, -5)
    assert result["vsx_status"].startswith("ERROR:")
    assert DV.classify_fate(result)[0] == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"


@pytest.mark.parametrize("row", [
    {"vsx_type": "EA"},
    {"vsx_type": "EA", "vsx_association_status": VA.SUPPORTED},
])
def test_legacy_or_status_only_type_never_promotes(row):
    assert DV.classify_fate(row)[0] == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"


def test_classifier_recomputes_native_type_and_rejects_incoherent_bindings():
    result = synthetic_association()
    result["vsx_type"] = "UXOR"
    assert DV.classify_fate(result)[0] == "KNOWN_ECLIPSING_BINARY(VSX:EA)"
    result["source_id"] = "9007199254740993"
    assert DV.classify_fate(result)[0] == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"


def test_classifier_recomputes_position_not_serialized_status():
    result = deepcopy(synthetic_association())
    result["vsx_candidates"][0]["RAJ2000"] = 11
    assert DV.classify_fate(result)[0] == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"


def test_confirmed_association_with_unavailable_type_keeps_distinct_gap():
    result = synthetic_association(Type="")
    fate, flags = DV.classify_fate(result)
    assert fate == "EVIDENCE_INCOMPLETE(VSX_CLASSIFICATION)"
    assert "vsx_type_unavailable" in flags
    assert "vsx_target_association_unresolved" not in flags


@pytest.mark.parametrize("other", [
    {"simbad_otype": "EB*"},
    {"ztf_class": "ZTF_ECLIPSING_PHASED", "ztf_period_d": 1.2},
])
def test_other_existing_mechanism_retains_vsx_gap_flag(other):
    fate, flags = DV.classify_fate({"vsx_type": "EA", **other})
    assert not fate.startswith("EVIDENCE_INCOMPLETE")
    assert "vsx_target_association_unresolved" in flags


@pytest.mark.parametrize("source_id", [77, 1000000000000000077])
def test_stage_default_uses_available_astrometry_without_extra_queries(tmp_path, monkeypatch, source_id):
    from seti.parallax4 import run as R

    pd.DataFrame([{
        "source_id": source_id, "vet_class": "SURVIVES", "kind": "DIP", "t_peak": 1700.3,
        "score": 9, "ra": 10.0, "dec": -5.0, "pmra": 1000.0, "pmdec": 0.0,
    }]).to_csv(tmp_path / "vetted.csv", index=False)
    queries = []
    def vsx_query(url, query):
        queries.append(query)
        return pd.DataFrame([native_row()])
    def offline(*args, **kwargs):
        raise RuntimeError("no acquisition in this synthetic control")
    monkeypatch.setattr(DV, "_tap_rows", vsx_query)
    report = R.stage_deepvet(
        {}, tmp_path, http=offline, tap=offline,
        simbad=lambda *a: {"simbad_status": "NO_MATCH"},
        ztf_fetch=lambda *a: pd.DataFrame(),
    )
    assert len(queries) == 1
    assert "SELECT TOP 101" in queries[0]
    assert report["fates"] == {"EVIDENCE_INCOMPLETE": 1}
    assert report["n_evidence_incomplete"] == 1 and report["unexplained"] == []
    text = (tmp_path / "deepvet.csv").read_text()
    assert "CALLER_REVIEWED_RECEIPT_MISSING" in text
    assert str(source_id) in text


def test_receipt_tolerance_larger_than_query_cannot_claim_unique_match(monkeypatch):
    monkeypatch.setattr(DV, "_tap_rows", lambda *a, **kw: pd.DataFrame([native_row()]))
    result = DV.vsx_type(10, -5, r_arcsec=1, target_astrometry=target(),
                        association_receipt=receipt(max_separation_arcsec=2))
    assert result["vsx_association_status"] == "MATCH_TOLERANCE_EXCEEDS_QUERY_RADIUS"
    assert "vsx_type" not in result
    assert DV.classify_fate(result)[0] == "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)"

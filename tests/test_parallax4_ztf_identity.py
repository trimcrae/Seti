"""ZTF catalogue identity is a precondition to a PARALLAX4 eclipse test.

Synthetic positive controls establish algorithm behavior only; no sky detection
or Gaia/ZTF counterpart identity is asserted by this suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.parallax4 import deepvet as DV


def quiet_mixed_cone():
    """Two individually constant objects whose observation windows mimic an EB."""
    rng = np.random.default_rng(81)
    mjd = np.sort(rng.uniform(58200, 60000, 600))
    phase = ((mjd - DV.GAIA_T0_MJD - 1700.3) / 2.345 + 0.5) % 1 - 0.5
    neighbor = np.abs(phase) < 0.05
    return pd.DataFrame({
        "mjd": mjd, "mag": np.where(neighbor, 15.6, 15.0),
        "magerr": 0.01, "filtercode": "zr",
        "oid": np.where(neighbor, "9007199254740993", "9007199254740992"),
    })


def forbid_bls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("ambiguous catalogue objects reached the eclipse fit")
    monkeypatch.setattr("astropy.timeseries.BoxLeastSquares", forbidden)


def test_quiet_mixed_object_cone_never_reaches_bls(monkeypatch):
    z = quiet_mixed_cone()
    assert z.groupby("oid")["mag"].nunique().to_list() == [1, 1]
    forbid_bls(monkeypatch)
    result = DV.ztf_eclipse_test(z, [1700.3 + 2.345 * k for k in (10, 40, 77, 120, 200)])
    assert result["ztf_class"] == "ZTF_AMBIGUOUS_OBJECTS"
    assert result["ztf_identity_status"] == "MULTIPLE_OBJECT_IDS"
    assert result["ztf_n_objects"] == 2 and result["ztf_n_band"] == 600
    assert result["ztf_object_ids"] == ["9007199254740992", "9007199254740993"]
    assert "ztf_period_d" not in result
    fate, flags = DV.classify_fate(result)
    assert fate == "EVIDENCE_INCOMPLETE(ZTF_OBJECT_IDENTITY)"
    assert "ztf_object_identity_unresolved" in flags


@pytest.mark.parametrize("bad_id", [
    None, pd.NA, "", " ", "NaN", "1e16", "1.0", "-1", "0",
    True, False, 9007199254740992.0, np.nan,
])
def test_missing_or_lossy_identifier_refuses_fit(monkeypatch, bad_id):
    z = quiet_mixed_cone().assign(oid="9007199254740993")
    z["oid"] = z["oid"].astype(object)
    z.loc[0, "oid"] = bad_id
    forbid_bls(monkeypatch)
    result = DV.ztf_eclipse_test(z, [1700.3])
    assert result["ztf_class"] == "ZTF_IDENTITY_UNRESOLVED"
    assert result["ztf_identity_status"] == "MALFORMED_OBJECT_ID"
    assert result["ztf_n_invalid_ids"] == 1
    assert "ztf_period_d" not in result


def test_missing_identifier_column_refuses_fit(monkeypatch):
    forbid_bls(monkeypatch)
    result = DV.ztf_eclipse_test(quiet_mixed_cone().drop(columns="oid"), [1700.3])
    assert result["ztf_class"] == "ZTF_IDENTITY_UNRESOLVED"
    assert result["ztf_identity_status"] == "MISSING_OBJECT_ID"
    assert result["ztf_n_invalid_ids"] == 600


def test_csv_preserves_adjacent_large_ids_and_null(monkeypatch):
    body = (b"oid,mjd,mag,magerr,filtercode,catflags\n"
            b"9007199254740992,59000,15,0.01,zr,0\n"
            b"9007199254740993,59001,15,0.01,zr,0\n"
            b",59002,15,0.01,zr,0\n"
            b"9007199254740994,59003,15,0.01,zr,1\n")
    z = DV.fetch_ztf(100, 20, http=lambda *a, **kw: (200, body))
    assert z["oid"].iloc[:2].to_list() == ["9007199254740992", "9007199254740993"]
    assert pd.isna(z["oid"].iloc[2]) and len(z) == 3
    forbid_bls(monkeypatch)
    result = DV.ztf_eclipse_test(z, [1700.3], min_points=1)
    assert result["ztf_class"] == "ZTF_IDENTITY_UNRESOLVED"
    assert result["ztf_n_objects"] == 2
    assert result["ztf_n_invalid_ids"] == 1


def test_single_integer_id_keeps_exact_provenance():
    z = quiet_mixed_cone().assign(oid=np.int64(9007199254740993), mag=15.0)
    result = DV.ztf_eclipse_test(z, [1700.3])
    assert result["ztf_identity_status"] == "SINGLE_OBJECT_ID"
    assert result["ztf_object_id"] == "9007199254740993"
    assert result["ztf_target_association"] == "NOT_ESTABLISHED_BY_OBJECT_ID"
    assert result["ztf_class"] == "ZTF_QUIET"


def test_band_specific_ids_do_not_pool_or_create_false_ambiguity():
    z = quiet_mixed_cone().assign(oid="9007199254740993", mag=15.0)
    other = z.head(100).assign(filtercode="zg", oid="1234567890", mag=17.0)
    result = DV.ztf_eclipse_test(pd.concat([z, other], ignore_index=True), [1700.3])
    assert result["ztf_identity_status"] == "SINGLE_OBJECT_ID"
    assert result["ztf_n"] == 700 and result["ztf_n_band"] == 600
    assert result["ztf_band"] == "zr"
    assert result["ztf_class"] == "ZTF_QUIET"


def test_other_catalogue_explanation_retains_identity_gap_flag():
    result = {"ztf_class": "ZTF_AMBIGUOUS_OBJECTS", "vsx_type": "EA"}
    fate, flags = DV.classify_fate(result)
    assert fate == "KNOWN_ECLIPSING_BINARY(VSX:EA)"
    assert "ztf_object_identity_unresolved" in flags


def test_stage_counts_identity_gap_separately(tmp_path, monkeypatch):
    from seti.parallax4 import run as R

    pd.DataFrame([{
        "source_id": 77, "vet_class": "SURVIVES", "flags": "", "kind": "DIP",
        "t_peak": 1700.3, "score": 9.0, "ra": 10.0, "dec": -5.0,
    }]).to_csv(tmp_path / "vetted.csv", index=False)
    def offline(*args, **kwargs):
        raise RuntimeError("offline control has no archive access")
    forbid_bls(monkeypatch)
    report = R.stage_deepvet(
        {}, tmp_path, http=offline, tap=offline,
        simbad=lambda ra, dec: {"simbad_status": "NO_MATCH"},
        vsx=lambda ra, dec: {"vsx_status": "NO_MATCH"},
        ztf_fetch=lambda ra, dec: quiet_mixed_cone(),
    )
    assert report["fates"] == {"EVIDENCE_INCOMPLETE": 1}
    assert report["n_evidence_incomplete"] == 1
    assert report["unexplained"] == []
    assert "1 evidence-incomplete" in report["verdict"]
    rows = pd.read_csv(tmp_path / "deepvet.csv")
    assert rows.loc[0, "ztf_identity_status"] == "MULTIPLE_OBJECT_IDS"
    assert "ztf_object_identity_unresolved" in rows.loc[0, "fate_flags"]

"""Offline tests for vetting promoted ZTF targets (``seti.tocsin.ztf_vet``) and for
the bright-neighbour rule and ledger prune it feeds (docs/tocsin-ztf.md 8d).

No network.  What is pinned: the analysis's verdicts on the shapes the two
false candidates had, the exclusion radius, the neighbour flagging, and that
removing a target from the ledger takes its trials with its events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.tocsin import targets as T
from seti.tocsin import ztf_vet as V
from seti.tocsin.ledger import Event, Ledger

SID = "4497414466452138496"
RA, DEC = 274.27374, 13.46533


def _target(**kw):
    t = {"source_id": SID, "ra": RA, "dec": DEC, "pmra": -20.0, "pmdec": -40.0,
         "phot_g_mean_mag": 16.0, "g_sdss_mag": 16.3178, "r_sdss_mag": 15.5796}
    t.update(kw)
    return t


def _dets(n=40, a_gaia=1.03, corrected_every=2, ref_mag=11.47, start=61200.0):
    """Detections whose difference flux is ``a_gaia`` times the target's Gaia flux,
    with every ``corrected_every``-th r detection carrying a reference of ``ref_mag``."""
    out = []
    for k in range(n):
        mjd = start + 2.0 * k
        for fid, base in ((1, 16.3178), (2, 15.5796)):
            dm = base - 2.5 * np.log10(a_gaia)
            corr = fid == 2 and corrected_every and (k % corrected_every == 0)
            out.append({"mjd": mjd, "fid": fid, "magpsf": dm, "isdiffpos": "t",
                        "corrected": corr, "magpsf_corr": ref_mag if corr else None,
                        "distnr": 0.9 if corr else 1.6, "drb": 0.9, "candid": f"{k}{fid}"})
    return out


def _obj(dets):
    o = {"oid": "ZTF18x", "meanra": RA, "meandec": DEC, "ndet": len(dets),
         "firstmjd": 58300.0, "lastmjd": 61297.0}
    return V.summarise_object(o, dets, [], _target(), RA, DEC)


def test_the_run_of_2026_09_20_shape_is_a_saturated_neighbour_residual():
    """A G 11.6 star 1" away, difference flux equal to the target's own flux, and a
    reference of r 11.5 wherever ALeRCE found one within 1.4": every flag fires
    and the classification names the mechanism."""
    neigh = [_target(), {"source_id": "999", "ra": RA + 1.0 / 3600 / np.cos(np.radians(DEC)),
                         "dec": DEC, "pmra": -20.0, "pmdec": -40.0, "phot_g_mean_mag": 11.6}]
    summ = _obj(_dets())
    rec = V.analyse(SID, _target(), 2026.55, neigh, [summ], [], [], [], 13.0, None)
    assert rec["classification"] == "systematic:saturated_neighbour_residual"
    for f in ("saturated_neighbour", "reference_brighter_than_target",
              "reference_saturated", "self_flux"):
        assert f in rec["flags"]
    assert rec["saturated_neighbour"]["source_id"] == "999"
    # The target itself is never its own neighbour.
    assert all(r["source_id"] != SID for r in rec["gaia_neighbours"]["rows"])
    # Both baselines were measured: a ~ 1 against Gaia, a ~ 2.4 % against the reference.
    assert summ["amplitude_vs_gaia"]["r"]["median"] == pytest.approx(1.03, abs=0.01)
    assert summ["amplitude_vs_reference"]["r"]["median"] == pytest.approx(0.024, abs=0.003)
    assert summ["reference_mag"]["r"]["median"] == pytest.approx(11.5, abs=0.02)
    assert summ["corrected_fraction"] == pytest.approx(0.25)   # half of r, none of g
    assert summ["distnr_arcsec"]["max"] == 1.6


def test_self_flux_alone_is_a_missing_reference_not_a_neighbour():
    """A high-proper-motion star absent from the reference: a ~ 1 in every band,
    no bright neighbour, no reference magnitude at all."""
    summ = _obj(_dets(a_gaia=1.0, corrected_every=0))
    rec = V.analyse(SID, _target(), 2026.55, [_target()], [summ], [], [], [], 13.0, None)
    assert rec["flags"] == ["self_flux"]
    assert rec["classification"] == "systematic:reference_missing_target"


def test_a_genuine_small_event_on_an_isolated_star_is_unexplained():
    dets = _dets(n=3, a_gaia=0.2, corrected_every=1, ref_mag=15.5)
    summ = _obj(dets)
    rec = V.analyse(SID, _target(), 2026.55, [_target()], [summ], [], [], [], 13.0, None)
    assert rec["flags"] == []
    assert rec["classification"] == "unexplained"


def test_known_variables_and_bright_dr_photometry_are_named():
    summ = _obj(_dets(n=3, a_gaia=0.2, corrected_every=1, ref_mag=15.5))
    simbad = [{"main_id": "V* XY Her", "otype": "EB*", "sep_arcsec": 0.4}]
    rec = V.analyse(SID, _target(), 2026.55, [], [summ], simbad, [], [], 13.0, None)
    assert "known_variable_simbad" in rec["flags"]
    assert rec["classification"] == "astrophysical:known_variable"
    vsx = [{"name": "XY Her", "type": "EA", "raj2000": RA, "dej2000": DEC}]
    rec = V.analyse(SID, _target(), 2026.55, [], [summ], [], vsx, [], 13.0, None)
    assert "known_variable_vsx" in rec["flags"]
    lc = [{"oid": "1", "mjd": 59000.0 + k, "mag": 11.5, "filtercode": "zr", "ra": RA, "dec": DEC}
          for k in range(12)]
    rec = V.analyse(SID, _target(), 2026.55, [], [summ], [], [], lc, 13.0, None)
    assert "dr_photometry_brighter_than_target" in rec["flags"]
    assert rec["classification"] == "systematic:blended_with_brighter_star"
    assert rec["irsa_lightcurve"]["objects"][0]["mag_median"] == 11.5


def test_irsa_csv_is_parsed_tolerantly():
    text = "\\ a comment line\noid,mjd,mag,magerr,filtercode\n1,59000.1,11.5,0.01,zr\n1,59001.1,11.6,0.01,zr\n"
    rows = V.parse_irsa_csv(text)
    assert len(rows) == 2 and rows[0]["filtercode"] == "zr"
    assert V.parse_irsa_csv("") == []


def test_the_queries_name_the_right_tables_and_fields():
    assert "gaiadr3.gaia_source" in V.gaia_cone_adql(RA, DEC, 60.0)
    assert "synthetic_photometry_gspc" in V.gaia_target_adql(SID)
    assert "FROM basic" in V.simbad_cone_adql(RA, DEC, 30.0)
    assert '"B/vsx/vsx"' in V.vsx_cone_adql(RA, DEC, 30.0)
    assert "FORMAT=csv" in V.irsa_lightcurve_url(RA, DEC, 3.0)


# ---------------------------------------------------------------------------
# the rule that acts on the finding: bright neighbours out of the target list
# ---------------------------------------------------------------------------
def test_exclusion_radius_scales_with_flux_and_is_zero_below_saturation():
    r = T.bright_neighbour_radius_arcsec([13.5, 13.0, 11.5, 8.0, 3.0], 13.0)
    assert r[0] == 0.0 and r[1] == 0.0
    assert r[2] == pytest.approx(5.0 * 10 ** 0.3)     # G 11.5 -> ~10"
    assert r[3] == pytest.approx(50.0)
    assert r[4] == 120.0                              # capped


def test_flag_bright_neighbours_names_the_offender_and_never_the_star_itself():
    cosd = np.cos(np.radians(5.0))
    targets = pd.DataFrame([
        {"source_id": "a", "ra": 10.0, "dec": 5.0, "phot_g_mean_mag": 16.0},
        {"source_id": "b", "ra": 20.0, "dec": 5.0, "phot_g_mean_mag": 16.0},
        {"source_id": "c", "ra": 30.0, "dec": 5.0, "phot_g_mean_mag": 12.5},
        {"source_id": "d", "ra": 20.0, "dec": 5.0 + 60.0 / 3600, "phot_g_mean_mag": 16.0},
    ])
    bright = pd.DataFrame([
        {"source_id": "n1", "ra": 10.0 + 1.0 / 3600 / cosd, "dec": 5.0, "phot_g_mean_mag": 11.5},
        {"source_id": "n2", "ra": 20.0, "dec": 5.0 + 40.0 / 3600, "phot_g_mean_mag": 8.0},
        {"source_id": "c", "ra": 30.0, "dec": 5.0, "phot_g_mean_mag": 12.5},
    ])
    out = T.flag_bright_neighbours(targets, bright, 13.0)
    by = {r.source_id: r for r in out.itertuples()}
    # a: 1" from a G 11.5 star (radius ~10").  b: 40" from a G 8 star (radius
    # 50").  d: 20" from the same G 8 star.  c is the bright star itself.
    assert set(by) == {"a", "b", "d"}
    assert by["a"].neighbour_source_id == "n1" and by["a"].sep_arcsec == pytest.approx(1.0, abs=0.01)
    assert by["b"].neighbour_source_id == "n2"


def test_flag_bright_neighbours_geometry():
    targets = pd.DataFrame([{"source_id": "d", "ra": 20.0, "dec": 5.0 + 60.0 / 3600,
                             "phot_g_mean_mag": 16.0},
                            {"source_id": "e", "ra": 20.0, "dec": 5.0 - 30.0 / 3600,
                             "phot_g_mean_mag": 16.0}])
    bright = pd.DataFrame([{"source_id": "n2", "ra": 20.0, "dec": 5.0 + 40.0 / 3600,
                            "phot_g_mean_mag": 8.0}])
    out = T.flag_bright_neighbours(targets, bright, 13.0)
    assert list(out["source_id"]) == ["d"]
    assert out.iloc[0]["sep_arcsec"] == pytest.approx(20.0, abs=0.01)
    assert out.iloc[0]["radius_arcsec"] == pytest.approx(50.0)
    assert T.flag_bright_neighbours(targets, bright.iloc[:0], 13.0).empty


def test_bright_star_adql_is_a_stripe_of_the_footprint():
    q = T.bright_star_adql(10.0, 20.0, -31.0, 90.0, 13.0)
    assert "phot_g_mean_mag < 13.0" in q and "ra >= 10.0 AND ra < 20.0" in q
    assert "BETWEEN -31.0 AND 90.0" in q


def _event(tid, night, mjd):
    return Event(target_id=tid, night=night, mjd=mjd, polarity="flash", bands=["g", "r"],
                 dflux_njy=1e6, dflux_err_njy=1e4, strongest_band="r", a=1.03, a_err=0.05,
                 sep_arcsec=0.1, sep_sigma=0.3, grey_z=0.1, grey_tested=True,
                 colour_temp_k=4700.0)


def test_remove_targets_takes_trials_with_events_and_is_recorded():
    led = Ledger()
    led.add_night("n1", [_event("x", "n1", 61235.3), _event("y", "n1", 61235.4)],
                  target_visits=1000, targets_in_footprint=1000, alerts_seen=50,
                  target_positions={"x": (274.27, 13.47), "y": (10.0, 5.0)},
                  bin_trials={"274,13": 100, "10,5": 100})
    led.add_night("n2", [_event("x", "n2", 61236.3)], target_visits=1000,
                  targets_in_footprint=1000, alerts_seen=50,
                  bin_trials={"274,13": 100, "10,5": 100})
    led.apply_visit_history({"x": [61235.3, 61236.3, 61237.3], "y": [61235.4]})
    assert led.n_events_kept == 3 and led.n_target_visits == 2000
    gone = led.remove_targets({"x": "bright_neighbour", "nobody": "whatever"})
    assert gone["events"] == 2 and gone["visits"] == 3
    assert "x" not in led.targets and "y" in led.targets
    assert led.n_events_kept == 1
    assert led.n_target_visits == 1997
    assert led.bin_trials["274,13"] == 197
    assert led.removed["x"]["reason"] == "bright_neighbour"
    # Idempotent, and a round trip through JSON keeps the record.
    assert led.remove_targets({"x": "again"})["events"] == 0
    import json
    led2 = Ledger(**{k: v for k, v in json.loads(json.dumps(led.__dict__)).items()
                     if k in Ledger.__dataclass_fields__})
    assert led2.removed["x"]["n_visits"] == 3
    led2.assess()
    assert led2.summary()["events_kept"] == 1

"""The aperture-scale contamination check over an EXISTING shortlist.

``stage_assess`` now runs the second, aperture-scale variability cone itself
(see ``run.stage_assess`` and ``vet.aperture_contamination``).  This stage
runs the same two cones -- the 3" identity cone and the aperture cone --
over the shortlist that the COMMITTED ``stars_vetted.csv`` already names,
without re-tiering the population.

Why a separate stage and not a re-assess.  MEASURED 2026-09-23: the only
screen shards whose artifacts survive for the committed shortlist (run
35652897914, 2026-09-21) predate the pool null; re-assessing them under the
current gauntlet crashed on the missing ``pn_n_trials`` column and, once that
is tolerated, would mark every star ``pool_null_unreached`` -- changing every
tier for a reason unrelated to contamination.  The aperture question is a
question about the sky around each shortlisted star, so it is asked of the
shortlist as it stands and its answer is written as a per-star record
(``aperture.json``) that :func:`seti.metronome.reconcile.reconcile_all`
lays over the assess tiers, exactly like the light-curve and vet records.
Demotion only ever removes a claim.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

#: the two vetoes this record can carry, most mundane first
APERTURE_VETOES = ("aperture_identity_periodic_variable", "aperture_contaminating_variable")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def aperture_veto(rec: dict | None) -> str | None:
    """The veto a star's aperture record carries, if any."""
    if not rec:
        return None
    if rec.get("identity_hit"):
        return "aperture_identity_periodic_variable"
    if rec.get("contaminating"):
        return "aperture_contaminating_variable"
    return None


def stage_aperture(conf: dict, out: Path, *, query_fn=None, cone_fn=None, mast_fn=None,
                   aperture_cone_fn=None, log=None, shortlist_keys=None) -> dict:
    """Identity + aperture cones for every star on the committed shortlist."""
    from .acquire import (
        AcquisitionLog,
        fetch_aperture_neighbours,
        fetch_positions,
        fetch_variable_context,
        tap_query,
    )
    from .run import _fill_positions_from_rotation, _load_rotation
    from .vet import DEFAULT_VET, aperture_contamination, periodic_variable

    out = Path(out)
    log = log or AcquisitionLog(prefix="metronome/aperture")
    vconf = dict(DEFAULT_VET, **(conf.get("vet") or {}))
    ap_conf = dict(conf.get("aperture_contamination") or {})
    vari = conf.get("variability_catalogues") or {}
    sources = set(vari)
    rep: dict = {"stage": "aperture", "generated_utc": _now(),
                 "radius_arcsec": ap_conf.get("radius_arcsec"),
                 "identity_radius_arcsec": float(conf.get("cone_radius_arcsec", 3.0)),
                 "sources": sorted(sources), "aperture_tol": float(vconf["aperture_tol"]),
                 "identity_tol": float(vconf["variable_tol"]),
                 "harmonics": list(vconf["variable_harmonics"])}

    vp = out / "stars_vetted.csv"
    if not vp.exists():
        rep["status"] = "NO_SHORTLIST"
        (out / "aperture.json").write_text(json.dumps(rep, indent=1, default=_jd))
        return rep
    df = pd.read_csv(vp, dtype={"star_key": str, "star_id": str})
    if shortlist_keys is not None:
        sl = df[df["star_key"].isin(set(shortlist_keys))]
    else:
        sl = df[df["fdr_watch"].astype(str).str.lower() == "true"]
    sl = sl.drop_duplicates("star_key")
    rep["shortlist_definition"] = ("stars_vetted.csv fdr_watch == True (every star at the "
                                   "watch FDR, whatever tier the gauntlet then gave it)")
    rep["n_shortlist"] = int(len(sl))
    rep["stars_vetted_tiers"] = sl["tier"].value_counts().to_dict()

    query_fn = query_fn or tap_query
    per: dict = {}
    reach_id: dict = {}
    reach_ap: dict = {}
    for mission in sorted(sl["mission"].dropna().astype(str).unique()):
        sub = sl[sl["mission"].astype(str) == mission]
        ids = sorted(set(sub["star_id"].astype(str)))
        pos = fetch_positions(ids, mission, query_fn=query_fn, mast_fn=mast_fn, log=log,
                              tables=conf.get("position_tables"))
        n_cat = len(pos)
        pos = _fill_positions_from_rotation(pos, ids, _load_rotation(out, mission))
        rep.setdefault("positions", {})[mission] = {
            "n_ids": len(ids), "n_from_kic_tic": int(n_cat), "n_total": int(len(pos))}
        v, rid = fetch_variable_context(pos, vari, cone_fn=cone_fn, log=log,
                                        radius_arcsec=float(conf.get("cone_radius_arcsec", 3.0)))
        a, rap = fetch_aperture_neighbours(
            pos, vari, radius_arcsec_by_mission=ap_conf.get("radius_arcsec") or {},
            mission=mission,
            identity_radius_arcsec=float(ap_conf.get("identity_radius_arcsec", 3.0)),
            cone_fn=aperture_cone_fn or cone_fn, log=log)
        posd = {str(r["star_id"]): (float(r["ra"]), float(r["dec"]))
                for _, r in pos.iterrows()}
        for _, r in sub.iterrows():
            sid, key = str(r["star_id"]), str(r["star_key"])
            period = float(r.get("period", np.nan))
            id_ok = sources <= rid.get(sid, set())
            ap_ok = sources <= rap.get(sid, set())
            reach_id[key], reach_ap[key] = id_ok, ap_ok
            ihit, idet = periodic_variable(period, v.get(sid, []), vconf["variable_harmonics"],
                                           float(vconf["variable_tol"]))
            chit, cdet = aperture_contamination(period, a.get(sid, []),
                                                vconf["variable_harmonics"],
                                                float(vconf["aperture_tol"]),
                                                vconf.get("aperture_period_range_days"))
            ra_dec = posd.get(sid)
            per[key] = {
                "star_id": sid, "mission": mission, "period": period,
                "tier_assess": r.get("tier"), "first_veto_assess": r.get("first_veto"),
                "catalogue": r.get("catalogue"),
                "ra": ra_dec[0] if ra_dec else None, "dec": ra_dec[1] if ra_dec else None,
                "identity_reached": bool(id_ok), "aperture_reached": bool(ap_ok),
                "identity_catalogued": [list(x) for x in v.get(sid, [])],
                "identity_hit": bool(ihit), "identity_detail": idet,
                "neighbours": a.get(sid, []),
                "contaminating": bool(chit), "contamination_detail": cdet,
            }
        # checkpoint after every mission: a job killed at its time limit keeps
        # what it measured (run 35869870343 lost everything at 180 minutes)
        rep["per_star"] = per
        rep["status"] = "PARTIAL"
        rep["missions_done"] = sorted(set(rep.get("missions_done", [])) | {mission})
        (out / "aperture.json").write_text(json.dumps(rep, indent=1, default=_jd))
        print(f"[metronome/aperture] {mission}: {len(sub)} stars, {len(pos)} positioned, "
              f"checkpoint written", flush=True)

    n = len(per)
    rep["n_positioned"] = int(sum(1 for p in per.values() if p["ra"] is not None))
    rep["frac_identity_reached"] = (sum(reach_id.values()) / n) if n else float("nan")
    rep["frac_aperture_reached"] = (sum(reach_ap.values()) / n) if n else float("nan")
    rep["per_star"] = per
    rep["flagged"] = sorted(k for k, p in per.items() if aperture_veto(p))
    rep["n_flagged"] = len(rep["flagged"])
    rep["flagged_detail"] = [
        {"star_key": k, "tier_assess": per[k]["tier_assess"], "veto": aperture_veto(per[k]),
         "period": per[k]["period"], "identity_detail": per[k]["identity_detail"],
         "matches": per[k]["contamination_detail"].get("matches"),
         "p_chance_any": per[k]["contamination_detail"].get("p_chance_any"),
         "n_distinct_with_period": per[k]["contamination_detail"].get("n_distinct_with_period")}
        for k in rep["flagged"]]
    rep["n_with_variable_neighbour"] = int(sum(1 for p in per.values() if p["neighbours"]))
    rep["acquisition"] = log.as_dict() if hasattr(log, "as_dict") else {}
    rep["status"] = "OK"
    rep["note"] = ("a contaminating hit is a CONTAMINATION flag -- a catalogued variable "
                   "neighbour inside the photometric aperture at the clock period or a low "
                   "harmonic -- not an identity flag; p_chance_any is the probability that "
                   "any of the star's distinct periodic neighbours would match by chance. "
                   "An identity hit is the 3\" cone the 2026-09-21 assess could not reach "
                   "for TESS stars (no positions)")
    (out / "aperture.json").write_text(json.dumps(rep, indent=1, default=_jd))

    from .reconcile import reconcile_all

    rep["reconciliation"] = reconcile_all(out, stage="aperture")
    (out / "aperture.json").write_text(json.dumps(rep, indent=1, default=_jd))
    print(f"[metronome/aperture] {n} shortlisted, {rep['n_positioned']} positioned, "
          f"identity reached {rep['frac_identity_reached']:.2f}, aperture reached "
          f"{rep['frac_aperture_reached']:.2f}; flagged: {rep['flagged']}")
    return rep


__all__ = ["APERTURE_VETOES", "aperture_veto", "stage_aperture"]

"""End-to-end donor / directed-travel analysis anchored on LHS 1140.

Hypothesis under test.  LHS 1140 b is a rocky, temperate, classical-habitable-zone
world (Rp ~ 1.73 R_earth) around a nearby (~14.96 pc) M dwarf.  *If* life arose
there, this channel asks the mirror of the K2-18 question -- not "who could have
seeded LHS 1140" but "whom could LHS 1140 have seeded":

1. **Recipients** (passive channel).  Resolve LHS 1140's full 6D phase space and
   reuse the K2-18 close-encounter engine with LHS 1140 as the anchor to rank the
   stars it passed CLOSE and SLOW to in the recent past -- the systems into which
   unbound ejecta could have been delivered.

2. **Destinations** (directed channel).  A technological disperser evolved on a
   rocky HZ world seeks OTHER rocky HZ worlds, so score the reachable neighbours
   with the **classical** (Earth-analog) destination prior -- the deliberate
   contrast with K2-18's hycean prior -- optionally layering the NASA Exoplanet
   Archive to flag known planet hosts and temperate (classical-HZ) worlds.

All of the maths is the reused, offline-tested :mod:`seti.panspermia` and
:mod:`seti.panspermia.reachability` machinery; only the anchor (LHS 1140) and the
habitability prior (classical) differ.  Acquisition is runner-side; outputs land in
``results/lhs1140_origin/``.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..panspermia.reachability import DEFAULT_SPEEDS_C, rank_targets
from ..panspermia.run import _anchor_phase_space, _resolve_anchor, panspermia_run

# LHS 1140 = Gaia DR3 2371032916186181760.  The runner resolves the live Gaia DR3
# 6D vector by source_id; these literature values (Cadieux et al. 2024) are a
# committed fallback only, used if acquisition fails, and are superseded by the
# fetched row.
LHS1140_SOURCE_ID = 2371032916186181760
LHS1140_FALLBACK = {
    "source_id": LHS1140_SOURCE_ID, "ra": 11.2487, "dec": -15.2742,
    "parallax": 66.83, "pmra": 317.6, "pmdec": -596.6,
    "radial_velocity": -13.7,   # km/s, systemic (literature); superseded by Gaia
}


def _resolve_lhs1140(source_id: int, anchor: dict | None) -> dict:
    """Resolve LHS 1140's raw 6D observables.

    Reuses the panspermia :func:`_resolve_anchor`, but that helper falls back to
    *K2-18* when the Gaia fetch fails; here we detect the wrong-anchor fallback and
    substitute the committed LHS 1140 literature vector instead, so the donor
    analysis is never silently run on the K2-18 anchor.
    """
    if anchor is not None:
        return dict(anchor)
    a = _resolve_anchor(source_id)
    if int(a.get("source_id", 0)) != LHS1140_SOURCE_ID:
        print("[lhs1140-origin] anchor resolve returned a non-LHS1140 row; "
              "using the committed LHS 1140 literature vector")
        return dict(LHS1140_FALLBACK)
    return a


def lhs1140_origin_run(cfg: Config | None = None, search_pc: float = 40.0,
                       t_max_myr: float = 10.0, source_id: int = LHS1140_SOURCE_ID,
                       g_max: float = 16.0, limit: int = 400000,
                       d_min_max_pc: float = 2.0, crossmatch: bool = False,
                       max_pc: float = 80.0, anchor: dict | None = None,
                       table: pd.DataFrame | None = None) -> dict:
    """Rank LHS 1140's recipients (passive) and its best destinations (directed).

    ``anchor``/``table`` may be supplied for offline tests instead of querying
    Gaia.  ``search_pc`` is the 3D radius around LHS 1140; ``t_max_myr`` the
    past-encounter viability window; ``d_min_max_pc`` the closest-approach cut
    defining the recipient shortlist.  With ``crossmatch`` the runner layers the
    NASA Exoplanet Archive onto the destination list (classical-HZ hosts).
    """
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "lhs1140_origin"
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor = _resolve_lhs1140(source_id, anchor)

    # 1. RECIPIENTS.  Reuse the K2-18 close-encounter engine with LHS 1140 as the
    # anchor to get the stars it passed close and slow to.  Route panspermia_run's
    # file output to a scratch root so it never clobbers the committed K2-18
    # results; we read the scored encounter table back and reframe it as a donor.
    scratch = copy.copy(cfg)
    scratch.root = Path(tempfile.mkdtemp(prefix="lhs1140_origin_"))
    try:
        panspermia_run(scratch, source_id=source_id, search_pc=search_pc,
                       g_max=g_max, limit=limit, t_max_myr=t_max_myr,
                       d_min_max_pc=d_min_max_pc, anchor=dict(anchor), table=table)
        enc = pd.read_csv(scratch.root / "results" / "panspermia" / "encounters_all.csv")
    finally:
        shutil.rmtree(scratch.root, ignore_errors=True)

    score = pd.to_numeric(enc.get("transfer_score", 0), errors="coerce").fillna(0.0)
    dmin = pd.to_numeric(enc.get("d_min_pc"), errors="coerce")
    recipients = enc[(score > 0) & (dmin <= d_min_max_pc)].sort_values(
        "transfer_score", ascending=False)
    comoving = (enc[enc["comoving"].astype(bool)].sort_values("v_rel_kms")
                if "comoving" in enc.columns else enc.iloc[:0])

    rec_cols = [c for c in ("source_id", "ra", "dec", "dist_pc", "phot_g_mean_mag",
                            "bp_rp", "radial_velocity", "sep_now_pc", "v_rel_kms",
                            "t_enc_myr", "d_min_pc", "transfer_score", "comoving")
                if c in enc.columns]
    if len(recipients):
        recipients[rec_cols].to_csv(out_dir / "recipients.csv", index=False)

    # 2. DESTINATIONS.  A traveller from a rocky HZ world seeks OTHER rocky HZ
    # worlds, so rank the reachable neighbours with the CLASSICAL (Earth-analog)
    # destination prior -- the deliberate contrast with K2-18's hycean prior.
    dest = rank_targets(enc, target="classical")
    xmatch_note = "not run (offline)"
    if crossmatch:
        try:
            from ..panspermia.exohosts import crossmatch_hosts, fetch_nearby_planets
            planets = fetch_nearby_planets(max_pc=max_pc)
            dest = crossmatch_hosts(dest, planets)
            # Boost hosts; boost classical-temperate (rocky-HZ) hosts most -- the
            # sharp signal for a rocky-world traveller (the mirror of the hycean
            # boost in the K2-18 channel).
            dest["dest_score"] = (dest["dest_score"]
                                  + 0.5 * dest["known_planet_host"].astype(float)
                                  + 1.0 * dest["has_temperate_planet"].astype(float))
            dest = dest.sort_values(["dest_score", "d_min_pc"], ascending=[False, True])
            xmatch_note = f"{len(planets)} archive planets < {max_pc} pc"
        except Exception as exc:  # noqa: BLE001
            xmatch_note = f"failed: {exc!r}"
            print(f"[lhs1140-origin] exoplanet cross-match {xmatch_note}")

    speed_cols = [f"cross_yr_{f:g}c" for f in DEFAULT_SPEEDS_C]
    dest_cols = [c for c in (["source_id", "ra", "dec", "dist_pc", "phot_g_mean_mag",
                              "bp_rp", "sep_now_pc", "v_rel_kms", "t_enc_myr",
                              "d_min_pc", "abs_g", "lum_class", "dest_score"]
                             + speed_cols
                             + ["known_planet_host", "n_planets", "has_temperate_planet",
                                "has_hycean_candidate", "host_name"])
                 if c in dest.columns]
    dest[dest_cols].to_csv(out_dir / "destinations.csv", index=False)

    # Anchor phase space, for an honest kinematic header in the summary.
    aps = _anchor_phase_space(dict(anchor))
    ms = dest[dest["lum_class"] == "main_sequence"] if "lum_class" in dest.columns else dest.iloc[:0]

    def _rec(r, cols) -> dict:
        return {k: (int(r[k]) if k == "source_id" else r[k])
                for k in cols if k in r and pd.notna(r.get(k))}

    rec_summary_cols = [c for c in ("source_id", "dist_pc", "bp_rp", "sep_now_pc",
                                    "v_rel_kms", "t_enc_myr", "d_min_pc",
                                    "transfer_score", "comoving") if c in enc.columns]
    dest_summary_cols = [c for c in ("source_id", "bp_rp", "dist_pc", "d_min_pc",
                                     "t_enc_myr", "lum_class", "dest_score",
                                     "host_name", "has_temperate_planet")
                         if c in dest.columns]

    summary = {
        "question": ("donor / directed-travel: which reachable worlds could an "
                     "LHS 1140 b biosphere have seeded, and which stars did LHS 1140 "
                     "pass near"),
        "anchor": {"name": "LHS 1140", "source_id": int(anchor.get("source_id", 0)),
                   "planet": "LHS 1140 b (rocky temperate HZ, Rp 1.73 R_earth)",
                   "dist_pc": round(float(aps["dist_pc"]), 3),
                   "U_kms": round(float(aps["U_kms"]), 3),
                   "V_kms": round(float(aps["V_kms"]), 3),
                   "W_kms": round(float(aps["W_kms"]), 3)},
        "destination_prior": "classical",
        "search_pc": search_pc, "t_max_myr": t_max_myr, "d_min_max_pc": d_min_max_pc,
        "crossmatch": xmatch_note,
        "n_searched": int(len(enc)),
        "n_recipients": int(len(recipients)),
        "n_comoving": int(len(comoving)),
        "closest_approach_pc": (round(float(recipients["d_min_pc"].min()), 4)
                                if len(recipients) else None),
        "top_recipients": [_rec(r, rec_summary_cols)
                           for _, r in recipients.head(20).iterrows()],
        "n_destinations": int((pd.to_numeric(dest.get("dest_score", 0),
                                             errors="coerce") > 0).sum()),
        "n_main_sequence": int(len(ms)),
        "n_known_hosts": int(dest.get("known_planet_host",
                                      pd.Series(dtype=bool)).sum()),
        "n_temperate_hosts": int(dest.get("has_temperate_planet",
                                          pd.Series(dtype=bool)).sum()),
        "top_destinations": [_rec(r, dest_summary_cols)
                             for _, r in dest.head(20).iterrows()],
        "notes": [
            "Linear (straight-line) closest approach is valid only over ~t_max_myr "
            "Myr; over longer baselines Galactic shear breaks the ballistic "
            "approximation -- use the galactic-encounters channel for a long-baseline "
            "donor variant.",
            "RV completeness: only Gaia DR3 sources WITH a radial velocity enter the "
            "6D search, so both lists are incomplete toward fainter neighbours.",
            "Fast flybys are reported but are non-capturable for the PASSIVE channel "
            "(relative speed far exceeds the reservoir escape speed); they still "
            "count as destinations for a DIRECTED traveller, which is unaffected by "
            "relative velocity.",
            "Co-moving neighbours (persistent low-v_rel companions) are labelled: "
            "they are the strongest passive bridge, not one-shot flybys.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[lhs1140-origin]", json.dumps({
        "n_searched": summary["n_searched"],
        "n_recipients": summary["n_recipients"],
        "n_comoving": summary["n_comoving"],
        "n_destinations": summary["n_destinations"],
        "n_temperate_hosts": summary["n_temperate_hosts"],
        "d_min_min_pc": summary["closest_approach_pc"]}))
    return summary


__all__ = ["lhs1140_origin_run", "LHS1140_SOURCE_ID", "LHS1140_FALLBACK"]

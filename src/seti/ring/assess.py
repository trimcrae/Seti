"""RING assessment: shortlist follow-up, the two-target vet, the verdict, the report.

The verdict is composed per host class and never reads as a sky statement
where a leg reached no data: a leg's ``NO_DATA_REACHED`` becomes a
``DEGRADED (...)`` prefix on the channel verdict, and the funnel counts are
reported per class so the census is legible host class by host class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ossuary import vet as ovet
from . import physics as ph
from . import screen as rscr
from .acquire import CATWISE_XMATCH_RENAME, normalise_xmatch, xmatch_at_epoch

_WD_HEADLINE = [
    "source_id", "wd_name", "ra", "dec", "l", "b", "dist_pc", "teff", "logg", "mass",
    "atmosphere", "pwd", "Gmag", "bp_rp", "Jmag", "Hmag", "Ksmag", "W1mag", "e_W1mag",
    "W2mag", "e_W2mag", "W3mag", "W4mag", "sed_anchor", "chi_W1", "chi_W2", "chi_color",
    "w1_w2_obs", "W1_excess_jy", "W2_excess_jy", "t_ring_k", "t_ring_lo_k", "t_ring_hi_k",
    "tau_ring", "tau_lo", "tau_hi", "ring_fit_chi2", "n_excess_bands", "ring_radius_au",
    "l_wd_w", "shape_class", "number_of_neighbours", "number_of_mates", "wise_angdist",
    "registration_arcsec", "chance_superposition_p", "gate_reason", "verdict",
    "ring_candidate", "f_min_300K_W2", "f_min_500K_W2", "f_min_700K_W2",
    "known_disk", "comovement_sigma", "comovement_ok", "blend_verdict", "n_beam_neighbours",
    "neighbour_over_excess", "simbad_id", "simbad_otype", "followup_verdict",
    "followup_reason",
]
_PSR_HEADLINE = [
    "jname", "bname", "ra", "dec", "pos_err_arcsec", "match_radius_arcsec", "pmra", "pmdec",
    "p0_s", "p1", "edot_w", "dist_kpc", "assoc", "binary", "bincomp", "ptype",
    "allwise_match", "allwise_dist_arcsec", "allwise_n_control_hits", "allwise_p_chance",
    "catwise_match", "catwise_dist_arcsec", "catwise_n_control_hits", "catwise_p_chance",
    "designation", "W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag", "W4mag", "ph_qual",
    "cc_flags", "catwise_name", "W1mag_cat", "W2mag_cat", "pmra_catwise", "pmdec_catwise",
    "w1_w2", "t_colour_k", "shape_class", "p_chance", "veto_reason", "verdict",
    "ring_candidate", "f_min_300K_W2", "f_min_500K_W2", "f_min_700K_W2",
    "ring_radius_300K_au", "ring_radius_500K_au", "ring_radius_700K_au",
    "localised", "colour_source", "w1_w2_err", "colour_secure", "e_W1mag_cat", "e_W2mag_cat",
    "p_chance_ring", "p_chance_any", "p_chance_trials", "allwise_p_chance_ring",
    "catwise_p_chance_ring",
    "allwise_n_control_ring_hits", "catwise_n_control_ring_hits",
    "allwise_n_sources_in_radius", "catwise_n_sources_in_radius",
    "allwise_n_sources_in_beam", "catwise_n_sources_in_beam",
]


# --------------------------------------------------------------------------
# White-dwarf shortlist follow-up (runner; every fetcher injectable)
# --------------------------------------------------------------------------

def gaia_neighbours_bulk(positions: pd.DataFrame, radius_arcsec: float = 12.0, *,
                         xmatch_fn=None) -> dict:
    """Gaia DR3 neighbours of every candidate in ONE CDS X-Match (I/355/gaiadr3).

    Run 36016037098 sent 27 per-object async Gaia-archive cones; every one
    hung past its 60 s allowance while the archive was dropping connections,
    so the blend test was untested for all 27.  CDS X-Match answers the whole
    list in one call.  Returns ``{candidate source_id (str): neighbour frame}``
    in the column names ``ovet.beam_blend_verdict`` reads.
    """
    from ..acquire.science import _xmatch

    xmatch_fn = xmatch_fn or _xmatch
    up = positions[["source_id", "ra", "dec"]].copy()
    up["source_id"] = up["source_id"].astype(str)
    raw = xmatch_fn(up, "vizier:I/355/gaiadr3", float(radius_arcsec))
    if raw is None or not len(raw):
        return {str(k): pd.DataFrame() for k in up["source_id"]}
    ren = {"Source": "nb_source_id", "RA_ICRS": "ra_nb", "DE_ICRS": "dec_nb",
           "Gmag": "phot_g_mean_mag", "BP-RP": "bp_rp", "pmRA": "pmra", "pmDE": "pmdec"}
    df = raw.rename(columns={k: v for k, v in ren.items() if k in raw.columns})
    out = {}
    for sid, g in df.groupby(df["source_id"].astype(str)):
        nb = pd.DataFrame({"source_id": g.get("nb_source_id", pd.Series(dtype=object))
                           .astype(str).to_numpy(),
                           "ra": pd.to_numeric(g.get("ra_nb"), errors="coerce").to_numpy(),
                           "dec": pd.to_numeric(g.get("dec_nb"), errors="coerce").to_numpy(),
                           "phot_g_mean_mag": pd.to_numeric(g.get("phot_g_mean_mag"),
                                                            errors="coerce").to_numpy(),
                           "bp_rp": pd.to_numeric(g.get("bp_rp"), errors="coerce").to_numpy()})
        out[sid] = nb[nb["source_id"] != sid].reset_index(drop=True)
    for k in up["source_id"]:
        out.setdefault(str(k), pd.DataFrame())
    return out


def wd_followup(shortlist: pd.DataFrame, cfg: dict, *, fetch_known_disks=None,
                fetch_neighbours=None, fetch_simbad=None, xmatch_fn=None,
                fetch_neighbours_bulk=None) -> pd.DataFrame:
    """Known-disk membership, CatWISE co-movement, beam blending, SIMBAD identity.

    Each service can fail; a failed test is recorded as untested, never as
    passed.  The follow-up verdict requires the co-movement test to have been
    made and passed, the beam to be clean or isolated, and no known-disk match.
    """
    if shortlist is None or not len(shortlist):
        return pd.DataFrame()
    out = shortlist.copy().reset_index(drop=True)
    c = cfg["contamination"]
    ep = cfg["epochs"]
    pos = out[["source_id", "ra", "dec"]].copy()
    # A Gaia source_id is an integer, but it arrives as a string from an
    # X-Match and as a float from a CSV round trip.  ``astype(..., errors=
    # "ignore")`` used to swallow the failure; pandas 3 removed that argument,
    # so the coercion is done explicitly and a non-numeric id is simply left
    # alone rather than taking the whole follow-up down with it.
    _sid = pd.to_numeric(pos["source_id"], errors="coerce")
    if _sid.notna().all():
        pos["source_id"] = _sid.astype("int64")

    out["known_disk"] = False
    if fetch_known_disks is not None:
        try:
            known = fetch_known_disks(pos)
            ids = pd.to_numeric(out["source_id"], errors="coerce")
            out["known_disk"] = ids.isin({int(k) for k in known
                                          if pd.notna(k)}).fillna(False)
        except Exception as exc:                        # noqa: BLE001
            print(f"[ring/wd] known-disk fetch failed: {exc!r}", flush=True)

    out["comovement_sigma"] = np.nan
    out["comovement_ok"] = False
    out["comovement_tested"] = False
    try:
        raw = xmatch_at_epoch(out[["source_id", "ra", "dec", "pmra", "pmdec"]],
                              "vizier:II/365/catwise", 3.0, float(ep["gaia"]),
                              float(ep["catwise"]), xmatch_fn=xmatch_fn)
        cat = normalise_xmatch(raw, CATWISE_XMATCH_RENAME)
        if len(cat):
            cat["source_id"] = cat["source_id"].astype(str)
            key = out["source_id"].astype(str)
            m = cat.set_index("source_id")
            for col in ("pmra_catwise", "pmdec_catwise", "e_pmra_catwise", "e_pmdec_catwise",
                        "W1mag_cat", "W2mag_cat", "catwise_name"):
                if col in m.columns:
                    out[col] = key.map(m[col]).to_numpy()
            if {"pmra_catwise", "pmdec_catwise"} <= set(out.columns):
                # Absent columns read as all-NaN Series, never a bare scalar
                # (run 36016037098: e_pmra_catwise absent -> "'numpy.float64'
                # object has no attribute 'fillna'" and 27/27 untested).
                e_r = rscr._numcol(out, "e_pmra_catwise").fillna(50.0)
                e_d = rscr._numcol(out, "e_pmdec_catwise").fillna(50.0)
                g_r = rscr._numcol(out, "pmra_error").fillna(1.0)
                g_d = rscr._numcol(out, "pmdec_error").fillna(1.0)
                d_r = (pd.to_numeric(out["pmra"], errors="coerce")
                       - pd.to_numeric(out["pmra_catwise"], errors="coerce")) / np.hypot(e_r, g_r)
                d_d = (pd.to_numeric(out["pmdec"], errors="coerce")
                       - pd.to_numeric(out["pmdec_catwise"], errors="coerce")) / np.hypot(e_d, g_d)
                sig = np.hypot(d_r, d_d)
                out["comovement_sigma"] = sig
                out["comovement_tested"] = sig.notna()
                out["comovement_ok"] = (sig <= float(c["astrometry"]["pm_consistency_sigma_max"])
                                        ).fillna(False)
    except Exception as exc:                            # noqa: BLE001
        print(f"[ring/wd] CatWISE co-movement failed: {exc!r}", flush=True)

    rows = []
    # Bounded: each Gaia cone gets ``per_call_s`` and the whole loop ``budget_s``.
    # Run 35992172700's assess job sat for hours in this loop -- an async Gaia
    # job that never returns blocks forever -- which would have cost the whole
    # run its summary.  A cone that times out, or is never reached, is recorded
    # as untested (``blend_untested``), never as passed.
    import concurrent.futures as _cf
    import time as _time

    fcfg = cfg.get("wd", {})
    per_call = float(fcfg.get("followup_per_call_s", 60.0))
    budget = float(fcfg.get("followup_budget_s", 2400.0))
    t_end = _time.monotonic() + budget
    pool = _cf.ThreadPoolExecutor(max_workers=int(fcfg.get("followup_workers", 4)))
    futs = {}
    bulk = None
    if fetch_neighbours_bulk is not None:
        try:
            bf = pool.submit(fetch_neighbours_bulk, out[["source_id", "ra", "dec"]])
            bulk = bf.result(timeout=min(budget, 900.0))
        except Exception as exc:                        # noqa: BLE001
            print(f"[ring/wd] bulk neighbour X-Match failed ({exc!r}); per-object cones",
                  flush=True)
            bulk = None
    if bulk is not None:
        class _Done:
            def __init__(self, v):
                self.v = v

            def result(self, timeout=None):
                return self.v
        for i, r in out.iterrows():
            futs[i] = _Done(bulk.get(str(r["source_id"]), pd.DataFrame()))
        fetch_neighbours = fetch_neighbours or (lambda ra, dec: None)
    elif fetch_neighbours is not None:
        for i, r in out.iterrows():
            futs[i] = pool.submit(fetch_neighbours, float(r["ra"]), float(r["dec"]))
    n_timeout = 0
    for i, r in out.iterrows():
        cand = r.to_dict()
        nb = None
        tested = fetch_neighbours is not None
        if fetch_neighbours is not None:
            left = t_end - _time.monotonic()
            try:
                if left <= 0:
                    raise TimeoutError("follow-up budget spent")
                nb = futs[i].result(timeout=min(per_call, left))
                if nb is not None and len(nb) and "source_id" in nb.columns:
                    nb = nb[nb["source_id"].astype(str) != str(cand["source_id"])]
            except Exception as exc:                    # noqa: BLE001
                tested = False
                n_timeout += isinstance(exc, (TimeoutError, _cf.TimeoutError))
                print(f"[ring/wd] neighbour fetch failed for {cand.get('source_id')}: "
                      f"{exc!r}", flush=True)
        rec = ovet.beam_blend_verdict(cand, nb, c)
        if fetch_neighbours is not None and not tested:
            rec["blend_verdict"] = "untested"
        rows.append(rec)
    pool.shutdown(wait=False, cancel_futures=True)
    out.attrs["n_neighbour_timeouts"] = n_timeout
    for k in rows[0]:
        out[k] = [r[k] for r in rows]
    if fetch_neighbours is None:
        out["blend_verdict"] = "untested"

    out["simbad_id"] = ""
    out["simbad_otype"] = ""
    if fetch_simbad is not None:
        try:
            _p = _cf.ThreadPoolExecutor(max_workers=1)
            sb = _p.submit(fetch_simbad, out[["source_id", "ra", "dec"]]).result(
                timeout=float(fcfg.get("followup_simbad_budget_s", 900.0)))
            _p.shutdown(wait=False, cancel_futures=True)
            if sb is not None and len(sb):
                sb = sb.copy()
                sb["source_id"] = sb["source_id"].astype(str)
                m = sb.set_index("source_id")
                key = out["source_id"].astype(str)
                out["simbad_id"] = key.map(m.get("simbad_id", "")).fillna("").to_numpy()
                out["simbad_otype"] = key.map(m.get("simbad_otype", "")).fillna("").to_numpy()
        except Exception as exc:                        # noqa: BLE001
            print(f"[ring/wd] SIMBAD failed: {exc!r}", flush=True)

    reason = pd.Series("", index=out.index, dtype=object)
    reason[out["known_disk"].astype(bool)] = "known_debris_disk"
    f = ~out["comovement_tested"].astype(bool) & (reason == "")
    reason[f] = "comovement_untested"
    f = out["comovement_tested"].astype(bool) & ~out["comovement_ok"].astype(bool) & \
        (reason == "")
    reason[f] = "background_source_no_comovement"
    f = ~out["blend_verdict"].isin(["clean", "isolated"]) & (reason == "")
    untested = out["blend_verdict"] == "untested"
    reason[f & untested] = "blend_untested"
    reason[f & ~untested] = "beam_blend"
    otype = rscr.text_column(out, "simbad_otype").str.lower()
    f = otype.str.contains("agn|qso|galaxy|seyfert|cv|nova|\\*\\*|sb|eb",
                           regex=True).fillna(False).astype(bool) & (reason == "")
    reason[f] = "simbad_identity"
    out["followup_reason"] = reason
    out["followup_verdict"] = np.where(reason == "", "surviving", "rejected")
    return out


# --------------------------------------------------------------------------
# The two-target pulsar vet
# --------------------------------------------------------------------------

def two_target_vet(psr_screen: pd.DataFrame, cfg: dict) -> list[dict]:
    out = []
    for tgt in cfg["pulsar"]["two_target_vet"]:
        rec = {"jname": tgt["jname"], "bname": tgt.get("bname", ""), "note": tgt.get("note", "")}
        if psr_screen is None or not len(psr_screen):
            rec["status"] = "NO_DATA_REACHED"
            out.append(rec)
            continue
        row = psr_screen[psr_screen["jname"] == tgt["jname"]]
        if not len(row):
            rec["status"] = "NOT_IN_CATALOGUE"
            out.append(rec)
            continue
        r = row.iloc[0]
        rec["status"] = "VETTED"
        for k in ("allwise_match", "allwise_dist_arcsec", "allwise_n_control_hits",
                  "allwise_p_chance", "catwise_match", "catwise_dist_arcsec",
                  "catwise_n_control_hits", "catwise_p_chance", "W1mag", "W2mag", "W1mag_cat",
                  "W2mag_cat", "w1_w2", "t_colour_k", "shape_class", "veto_reason", "verdict",
                  "edot_w", "dist_kpc", "match_radius_arcsec", "pos_err_arcsec",
                  "f_min_300K_W2", "f_min_500K_W2", "f_min_700K_W2", "ring_radius_500K_au",
                  "assoc", "binary", "bincomp"):
            if k in r.index:
                v = r[k]
                rec[k] = (None if (isinstance(v, float) and not np.isfinite(v)) else
                          (v.item() if hasattr(v, "item") else v))
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Verdict and report
# --------------------------------------------------------------------------

def compose_verdict(legs: dict) -> tuple[str, list[str]]:
    """One channel verdict from the four leg summaries."""
    degraded = [k for k, s in legs.items()
                if not s or s.get("status") in (None, "NO_DATA_REACHED", "QUERY_FAILED",
                                                 "NOT_RUN", "NO_TESTABLE_OBJECTS",
                                                 "INSUFFICIENT_EPOCHS")]
    partial = [k for k, s in legs.items() if k not in degraded and s
               and s.get("coverage_degraded")]
    reached = [k for k in legs if k not in degraded]
    if not reached:
        return "NO_DATA_REACHED", degraded + partial
    wd = legs.get("wd") or {}
    psr = legs.get("pulsar") or {}
    bd = legs.get("bd") or {}
    ffp = legs.get("ffp") or {}
    n_wd = int(wd.get("n_ring_candidates_after_followup", wd.get("n_ring_candidates", 0)) or 0)
    n_psr = int(psr.get("n_ring_candidates", 0) or 0)
    n_bd = int(bd.get("n_duty_cycle_flags", 0) or 0)
    n_ffp = int(ffp.get("n_flags_planetary_and_hot", 0) or 0)
    if n_wd + n_psr > 0:
        v = "RING_CANDIDATES_PENDING_VET"
    elif n_bd + n_ffp > 0:
        v = "NO_RING_SURVIVOR; SECONDARY_FLAGS_PENDING_VET"
    else:
        v = "NO_RING_SURVIVOR"
    notes = []
    if degraded:
        notes.append(f"{', '.join(degraded)} not reached")
    if partial:
        notes.append("; ".join(f"{k} partial: {legs[k]['coverage_degraded']}" for k in partial))
    if notes:
        v = f"DEGRADED ({'; '.join(notes)}); {v}"
    return v, degraded + partial


def report_md(summary: dict) -> str:
    L = ["# RING — rings around the dead (S63)", "",
         f"**Verdict:** `{summary.get('verdict')}`", "",
         f"Generated {summary.get('generated_at')} by run `{summary.get('run_id')}`. "
         "Leg provenance (the run whose files each leg's numbers come from): "
         + ", ".join(f"{k}: `{(v or {}).get('run_id')}` ({(v or {}).get('generated_at')})"
                     for k, v in (summary.get("leg_provenance") or {}).items()), ""]
    cons = summary.get("consistency") or {}
    if cons.get("failures"):
        L += ["**Self-consistency failures:** " + "; ".join(cons["failures"]), ""]
    legs = summary.get("legs", {})
    wd = legs.get("wd") or {}
    if wd:
        L += ["## White dwarfs", "",
              f"* input hosts with an AllWISE counterpart: {wd.get('n_input', 0):,} "
              f"(route: `{summary.get('acquisition', {}).get('wd', {}).get('route')}`)",
              f"* SED anchor: {wd.get('sed_anchor_counts')}",
              f"* W1 / W2 detections: {wd.get('n_w1_detected', 0):,} / "
              f"{wd.get('n_w2_detected', 0):,}",
              f"* excess-flagged: {wd.get('n_excess_flagged', 0):,}; shapes: "
              f"{wd.get('shape_counts')}",
              f"* surviving the catalogue gates: {wd.get('n_surviving_gates', 0):,}; "
              f"gate rejections: {wd.get('gate_reasons')}",
              f"* ring-band candidates before follow-up: {wd.get('n_ring_candidates', 0):,}",
              f"* after follow-up: **{wd.get('n_ring_candidates_after_followup', 'n/a')}** "
              f"({wd.get('followup_reasons')})",
              f"* sensitivity: {wd.get('sensitivity')}",
              f"* registration test evaluated for {wd.get('n_registration_tested', 'n/a')} "
              f"hosts; ring-band gate reasons: {wd.get('ring_band_gate_reasons')}",
              f"* chance census (offset-position controls): {wd.get('chance_census')}",
              f"* mechanism per flagged excess: {wd.get('mechanism_counts')}; ring vetoes: "
              f"{wd.get('ring_vetoes')}", ""]
    psr = legs.get("pulsar") or {}
    if psr:
        L += ["## Pulsars", "",
              f"* hosts: {psr.get('n_hosts', 0):,} "
              f"(route: `{summary.get('acquisition', {}).get('pulsar', {}).get('route')}`)",
              f"* with an AllWISE / CatWISE counterpart inside the per-pulsar radius: "
              f"{psr.get('n_with_allwise_counterpart', 0):,} / "
              f"{psr.get('n_with_catwise_counterpart', 0):,} "
              f"(any: {psr.get('n_with_any_counterpart', 0):,})",
              f"* median control hits per host over {psr.get('control_positions_per_host')} "
              f"offset positions: {psr.get('median_allwise_control_hits')}",
              f"* counterpart shapes: {psr.get('shape_counts')}",
              f"* vetoes: {psr.get('veto_reasons')}",
              f"* surviving: {psr.get('n_surviving', 0):,}; ring-band: "
              f"**{psr.get('n_ring_candidates', 0):,}**",
              f"* localised hosts (2 x position error <= the widest aperture): "
              f"{psr.get('n_localised', 'n/a')}",
              f"* chance census (observed vs the local-control expectation): "
              f"{psr.get('chance_census')}",
              f"* sensitivity: {psr.get('sensitivity')}", ""]
        fates = psr.get("ring_band_fates") or []
        if fates:
            L += ["### Every ring-band-coloured counterpart and its fate", "",
                  "| pulsar | localised | pos err (\") | colour src | W1-W2 | T (K) | "
                  "p_chance(ring) | veto |", "|---|---|---|---|---|---|---|---|"]
            for f in fates:
                def _f(x, fmt):
                    return "n/a" if x is None else format(x, fmt)
                L.append(f"| {f.get('jname')} | {f.get('localised')} | "
                         f"{_f(f.get('pos_err_arcsec'), '.3g')} | {f.get('colour_source')} | "
                         f"{_f(f.get('w1_w2'), '.2f')} +/- {_f(f.get('w1_w2_err'), '.2f')} | "
                         f"{_f(f.get('t_colour_k'), '.0f')} | {_f(f.get('p_chance_ring'), '.2g')} | "
                         f"{f.get('veto_reason') or 'SURVIVING'} |")
            L.append("")
        tv = summary.get("two_target_vet", [])
        if tv:
            L += ["### The two pulsar-planet systems", ""]
            for t in tv:
                L.append(f"* **{t.get('bname') or t.get('jname')}** ({t.get('jname')}): "
                         f"{t.get('status')}; AllWISE match {t.get('allwise_match')} "
                         f"(controls {t.get('allwise_n_control_hits')}, p {t.get('allwise_p_chance')}); "
                         f"CatWISE match {t.get('catwise_match')}; shape `{t.get('shape_class')}`; "
                         f"verdict `{t.get('verdict')}` {t.get('veto_reason') or ''}; "
                         f"f_min(500 K, W2) {t.get('f_min_500K_W2')}; {t.get('note')}")
            L.append("")
    bd = legs.get("bd") or {}
    if bd:
        L += ["## Brown dwarfs (Y / late T): W2 duty cycle", "",
              f"* targets: {bd.get('n_targets', 0):,}; with NEOWISE epochs: "
              f"{bd.get('n_with_epochs', 0):,}; tested (>= min epochs): {bd.get('n_tested', 0):,}",
              f"* population W2 reduced-chi2 median: {bd.get('population_w2_chi2_median')}; "
              f"threshold {bd.get('w2_chi2_threshold')}",
              f"* duty-cycle flags: **{bd.get('n_duty_cycle_flags', 0):,}**",
              f"* coverage: {bd.get('epoch_coverage')}"
              + (f" -- DEGRADED: {bd['coverage_degraded']}" if bd.get("coverage_degraded")
                 else ""), ""]
        for f in bd.get("flagged", [])[:30]:
            L.append(f"  * `{f.get('source_id')}` {f.get('spt')}: n={f.get('w2_n_epochs')}, "
                     f"chi2_red={f.get('w2_chi2_red'):.1f}, amp={f.get('w2_amp_mag'):.2f} mag, "
                     f"high-state fraction {f.get('w2_duty_cycle_high')}")
        for f in bd.get("duty_cycle_vetoed", [])[:30]:
            L.append(f"  * vetoed `{f.get('source_id')}` {f.get('spt')}: chi2_red="
                     f"{f.get('w2_chi2_red')}, amp={f.get('w2_amp_mag')} mag, W1-W2="
                     f"{f.get('w1_w2_mean')}, pm_known={f.get('pm_known')} -> "
                     f"{f.get('duty_cycle_veto')}")
        L.append("")
    ffp = legs.get("ffp") or {}
    if ffp:
        L += ["## Free-floating planetary-mass objects: hotter than cooling allows", "",
              f"* with L_bol: {ffp.get('n_with_lbol')}; with a group: {ffp.get('n_with_group')}; "
              f"age sources: {ffp.get('age_source_counts')}; unmatched group names: "
              f"{ffp.get('unmatched_group_names')}",
              f"* objects: {ffp.get('n_objects', 0):,}; testable (L_bol and an age): "
              f"{ffp.get('n_testable', 0):,}; catalogued planetary-mass: "
              f"{ffp.get('n_catalogued_planetary_mass', 0):,}",
              f"* above the 13 M_J cooling ceiling (with the 0.5 dex margin): "
              f"{ffp.get('n_hotter_than_ceiling', 0):,}; of which catalogued planetary-mass: "
              f"**{ffp.get('n_flags_planetary_and_hot', 0):,}**",
              "* every flag carries `age_misassignment|mass_underestimate|unresolved_binary` "
              "as systematics not excluded", ""]
    L += ["See `docs/ring.md` for the claim, the prior art and the contamination model.", ""]
    return "\n".join(L)


def json_safe(x):
    """Strict-JSON copy: NaN/inf -> None, numpy scalars -> Python, recursively.

    ``json.dumps`` writes a Python float NaN as the bare token ``NaN`` (and inf
    as ``Infinity``) without consulting ``default``; the summary of run
    35752692549 carried both, which strict parsers reject.
    """
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if x is pd.NA or x is pd.NaT:
        return None
    return x


def consistency_checks(summary: dict) -> dict:
    """Do the summary's own counts add up?  Failures are listed, not raised."""
    fails, checked = [], 0
    legs = summary.get("legs") or {}
    psr = legs.get("pulsar") or {}
    if psr.get("status") == "OK":
        n_any = psr.get("n_with_any_counterpart")
        for name in ("shape_counts",):
            tot = sum((psr.get(name) or {}).values())
            checked += 1
            if tot != n_any:
                fails.append(f"pulsar {name} sum {tot} != n_with_any_counterpart {n_any}")
        vetoed = sum((psr.get("veto_reasons") or {}).values())
        checked += 1
        if vetoed + int(psr.get("n_surviving") or 0) != n_any:
            fails.append(f"pulsar vetoed {vetoed} + surviving {psr.get('n_surviving')} "
                         f"!= n_with_any_counterpart {n_any}")
        checked += 1
        if len(psr.get("survivors") or []) != min(50, int(psr.get("n_surviving") or 0)):
            fails.append("pulsar survivors list length != n_surviving")
        fates = psr.get("ring_band_fates")
        if fates is not None:
            checked += 1
            if len(fates) != int((psr.get("shape_counts") or {}).get("ring_band", 0)):
                fails.append("ring_band_fates length != shape_counts.ring_band")
    bd = legs.get("bd") or {}
    if bd.get("status") == "OK":
        checked += 1
        if not (int(bd.get("n_tested") or 0) <= int(bd.get("n_with_epochs") or 0)
                <= int(bd.get("n_targets") or 0)):
            fails.append("bd n_tested <= n_with_epochs <= n_targets violated")
        acq = (summary.get("acquisition") or {}).get("bd") or {}
        checked += 1
        if acq.get("status") in (None, "NOT_RUN") and not bd.get("coverage_degraded"):
            fails.append("bd screened OK but no acquisition record, and not marked degraded")
    ffp = legs.get("ffp") or {}
    if ffp.get("n_objects"):
        checked += 1
        if int(ffp.get("n_testable") or 0) > int(ffp.get("n_objects") or 0):
            fails.append("ffp n_testable > n_objects")
    gen = summary.get("generated_at")
    for k, v in (summary.get("leg_provenance") or {}).items():
        checked += 1
        if v.get("generated_at") and gen and v["generated_at"] > gen:
            fails.append(f"leg {k} generated after the summary")
    return {"n_checks": checked, "failures": fails, "ok": not fails}


def write_summary(out_dir: Path, summary: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = json_safe(summary)
    (out_dir / "summary.json").write_text(json.dumps(safe, indent=2, default=_json_default,
                                                     allow_nan=False))
    (out_dir / "REPORT.md").write_text(report_md(safe))


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


__all__ = ["wd_followup", "two_target_vet", "compose_verdict", "report_md", "write_summary",
           "json_safe", "consistency_checks",
           "_WD_HEADLINE", "_PSR_HEADLINE", "ph"]

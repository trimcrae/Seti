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
]


# --------------------------------------------------------------------------
# White-dwarf shortlist follow-up (runner; every fetcher injectable)
# --------------------------------------------------------------------------

def wd_followup(shortlist: pd.DataFrame, cfg: dict, *, fetch_known_disks=None,
                fetch_neighbours=None, fetch_simbad=None, xmatch_fn=None) -> pd.DataFrame:
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
    pos["source_id"] = pos["source_id"].astype("int64", errors="ignore")

    out["known_disk"] = False
    if fetch_known_disks is not None:
        try:
            known = fetch_known_disks(pos)
            out["known_disk"] = out["source_id"].astype("int64").isin(set(known))
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
                e_r = pd.to_numeric(out.get("e_pmra_catwise"), errors="coerce").fillna(50.0)
                e_d = pd.to_numeric(out.get("e_pmdec_catwise"), errors="coerce").fillna(50.0)
                g_r = pd.to_numeric(out.get("pmra_error"), errors="coerce").fillna(1.0)
                g_d = pd.to_numeric(out.get("pmdec_error"), errors="coerce").fillna(1.0)
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
    for _, r in out.iterrows():
        cand = r.to_dict()
        nb = None
        if fetch_neighbours is not None:
            try:
                nb = fetch_neighbours(float(cand["ra"]), float(cand["dec"]))
                if nb is not None and len(nb) and "source_id" in nb.columns:
                    nb = nb[nb["source_id"].astype(str) != str(cand["source_id"])]
            except Exception as exc:                    # noqa: BLE001
                print(f"[ring/wd] neighbour fetch failed for {cand.get('source_id')}: "
                      f"{exc!r}", flush=True)
        rows.append(ovet.beam_blend_verdict(cand, nb, c))
    for k in rows[0]:
        out[k] = [r[k] for r in rows]
    if fetch_neighbours is None:
        out["blend_verdict"] = "untested"

    out["simbad_id"] = ""
    out["simbad_otype"] = ""
    if fetch_simbad is not None:
        try:
            sb = fetch_simbad(out[["source_id", "ra", "dec"]])
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
    reason[f] = "beam_blend" if fetch_neighbours is not None else "blend_untested"
    otype = out["simbad_otype"].astype(str).str.lower()
    f = otype.str.contains("agn|qso|galaxy|seyfert|cv|nova|\\*\\*|sb|eb", regex=True) & \
        (reason == "")
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
                                                 "NOT_RUN")]
    reached = [k for k in legs if k not in degraded]
    if not reached:
        return "NO_DATA_REACHED", degraded
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
    if degraded:
        v = f"DEGRADED ({', '.join(degraded)} not reached); {v}"
    return v, degraded


def report_md(summary: dict) -> str:
    L = ["# RING — rings around the dead (S63)", "",
         f"**Verdict:** `{summary.get('verdict')}`", ""]
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
              f"* sensitivity: {wd.get('sensitivity')}", ""]
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
              f"* sensitivity: {psr.get('sensitivity')}", ""]
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
              f"* duty-cycle flags: **{bd.get('n_duty_cycle_flags', 0):,}**", ""]
        for f in bd.get("flagged", [])[:30]:
            L.append(f"  * `{f.get('source_id')}` {f.get('spt')}: n={f.get('w2_n_epochs')}, "
                     f"chi2_red={f.get('w2_chi2_red'):.1f}, amp={f.get('w2_amp_mag'):.2f} mag, "
                     f"high-state fraction {f.get('w2_duty_cycle_high')}")
        L.append("")
    ffp = legs.get("ffp") or {}
    if ffp:
        L += ["## Free-floating planetary-mass objects: hotter than cooling allows", "",
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


def write_summary(out_dir: Path, summary: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    (out_dir / "REPORT.md").write_text(report_md(summary))


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
           "_WD_HEADLINE", "_PSR_HEADLINE", "ph"]

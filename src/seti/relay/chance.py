"""RELAY chance model: how many pair-line drift matches does noise give?

The drift prior of ``docs/relay.md`` §3 is centred on the unremoved Earth term
and is as wide as the pipeline's resolution plus the small-angle leak -- the
Gaia kinematic part is 10^-8 Hz/s and contributes nothing.  So whether a hit
"matches" is decided almost entirely by how small its drift is, and the only
honest yardstick is the drift distribution the published hit lists actually
have.  This module prices every pair-line hit against that distribution:

* **drift null** -- for each pair-line hit i with window W_i (centre +/- k
  tight half-widths, exactly the window ``stage_assess`` used), p_i is the
  fraction of the OTHER valid hits whose drift falls inside W_i.  The expected
  number of matches is sum(p_i); the tail P(N >= observed) is the
  Poisson-binomial tail over the p_i.  A match count at sum(p_i) says the
  window selects nothing that any other hit would not also satisfy.
* **geometry null** -- the fraction of the in-sample BL targets that are
  transmitters of at least one qualifying pair at that beam.  When it is ~1
  the "on a pair line" condition selects nothing either.
* **hygiene** -- rows read from a column that is not a frequency (a
  "Frequency rank"), rows from an injection-recovery table, and rows that
  appear twice because two seeds resolved to the same e-print are flagged and
  excluded before anything is counted; frequencies inside a known terrestrial
  allocation and frequencies recurring on >= 3 unrelated sightlines within a
  wide bin are flagged as RFI.

``python -m seti.relay.chance`` recomputes all of it OFFLINE from the files a
run committed (``hits_crossmatch.csv``, ``hits.json``, ``summary.json``) and
checks those files describe the same run before it writes anything.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------
_NOT_A_FREQUENCY = re.compile(r"\brank\b|\bindex\b|\border\b|\bnumber\b|#", re.I)
_INJECTION = re.compile(r"inject|artificial signal|synthetic signal|simulated signal", re.I)


def _arxiv_of(table) -> str:
    s = str(table or "")
    m = re.match(r"arXiv:([^:]+):", s)
    return m.group(1) if m else s


def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def hit_hygiene(hits: pd.DataFrame) -> pd.DataFrame:
    """Flag parse artefacts and duplicate rows; returns a copy with the flags."""
    h = hits.copy()
    fh = h["freq_header"].astype(str) if "freq_header" in h else pd.Series("", index=h.index)
    cap = h["caption"].astype(str) if "caption" in h else pd.Series("", index=h.index)
    h["artefact_not_frequency"] = fh.str.contains(_NOT_A_FREQUENCY).to_numpy()
    h["artefact_injection_table"] = cap.str.contains(_INJECTION).to_numpy()
    arx = h["table"].map(_arxiv_of) if "table" in h else pd.Series("", index=h.index)
    tk = h["target"].map(_key) if "target" in h else pd.Series("", index=h.index)
    dkey = pd.DataFrame({"a": arx, "t": tk, "f": np.round(h["freq_mhz"].to_numpy(float), 6),
                         "d": np.round(h["drift_hz_s"].to_numpy(float), 6)}, index=h.index)
    h["duplicate_row"] = dkey.duplicated(keep="first").to_numpy()
    h["valid_hit"] = ~(h["artefact_not_frequency"] | h["artefact_injection_table"] | h["duplicate_row"])
    h["valid_hit"] &= np.isfinite(h["freq_mhz"].to_numpy(float)) & np.isfinite(h["drift_hz_s"].to_numpy(float))
    return h


def known_rfi_band(freq_mhz, bands) -> list:
    """Name of the terrestrial/satellite allocation containing each frequency, or None."""
    f = np.asarray(freq_mhz, float)
    out = [None] * len(f)
    for i, v in enumerate(f):
        if not np.isfinite(v):
            continue
        for b in bands or ():
            if float(b["lo"]) <= v <= float(b["hi"]):
                out[i] = str(b["name"])
                break
    return out


def wide_recurrence(hits: pd.DataFrame, width_khz: float, min_sightlines: int,
                    valid=None) -> np.ndarray:
    """True where >= min_sightlines distinct targets share a frequency within +/- width/2.

    A sliding window, not fixed bins, so a family straddling a bin edge is
    not split.  Only valid rows count as sightlines.
    """
    f = hits["freq_mhz"].to_numpy(float)
    t = hits["target"].map(_key).to_numpy() if "target" in hits else np.array([""] * len(f))
    ok = np.isfinite(f) if valid is None else (np.asarray(valid, bool) & np.isfinite(f))
    half = 0.5 * float(width_khz) * 1e-3
    out = np.zeros(len(f), bool)
    for i in np.flatnonzero(ok):
        near = ok & (np.abs(f - f[i]) <= half)
        out[i] = len(set(t[near])) >= int(min_sightlines)
    return out


# ---------------------------------------------------------------------------
# the chance model
# ---------------------------------------------------------------------------
def poisson_binomial_tail(p, k: int) -> float:
    """P(sum of independent Bernoulli(p_i) >= k)."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return float(dist[max(int(k), 0):].sum()) if k <= len(p) else 0.0


def drift_null(drifts_pool: np.ndarray, pool_index: np.ndarray, on_index: np.ndarray,
               drift_on: np.ndarray, centres: np.ndarray, halfwidths: np.ndarray,
               k_sig: float, rfi_on: np.ndarray | None = None,
               drift_range_hz_s: float = 4.0) -> dict:
    """Expected matches when a hit's drift is replaced by another valid hit's.

    ``pool_index``/``on_index`` identify rows so a hit is never compared with
    itself (leave-one-out).
    """
    drifts_pool = np.asarray(drifts_pool, float)
    pool_index = np.asarray(pool_index)
    n_on = len(on_index)
    ps, inside, fr = [], [], []
    for j in range(n_on):
        c, hw = float(centres[j]), float(halfwidths[j])
        lo, hi = c - k_sig * hw, c + k_sig * hw
        others = drifts_pool[pool_index != on_index[j]]
        ps.append(float(np.mean((others >= lo) & (others <= hi))) if len(others) else np.nan)
        inside.append(bool(np.isfinite(drift_on[j]) and lo <= drift_on[j] <= hi))
        fr.append(min(1.0, (hi - lo) / (2.0 * drift_range_hz_s)))
    ps = np.asarray(ps, float)
    inside = np.asarray(inside, bool)
    rfi_on = np.zeros(n_on, bool) if rfi_on is None else np.asarray(rfi_on, bool)
    n_match = int(inside.sum())
    n_cand = int((inside & ~rfi_on).sum())
    return {"n_trials": n_on,
            "n_drift_match": n_match,
            "n_expected_drift_match": round(float(np.nansum(ps)), 3),
            "p_value_drift_match": round(poisson_binomial_tail(ps, n_match), 4) if n_on else None,
            "n_candidates_after_rfi": n_cand,
            "n_expected_candidates": round(float(np.nansum(ps[~rfi_on])), 3),
            "p_value_candidates": round(poisson_binomial_tail(ps[~rfi_on], n_cand), 4) if n_on else None,
            "mean_window_fraction_of_hits": round(float(np.nanmean(ps)), 3) if np.isfinite(ps).any() else None,
            "mean_window_fraction_of_uniform_drift_range": round(float(np.mean(fr)), 4) if n_on else None,
            "uniform_drift_range_hz_s": drift_range_hz_s,
            "per_hit_p": [round(float(x), 3) for x in ps]}


def geometry_null(n_targets_as_transmitter, n_targets_in_sample) -> float | None:
    if not n_targets_in_sample or n_targets_as_transmitter is None:
        return None
    return round(float(n_targets_as_transmitter) / float(n_targets_in_sample), 4)


def assess_chance(hits: pd.DataFrame, beam_names, *, k_sig: float, rfi_col: str = "rfi_flag",
                  drift_range_hz_s: float = 4.0) -> dict:
    """Per-beam drift-null expectation over the VALID hits (``hit_hygiene`` first)."""
    h = hits if "valid_hit" in hits else hit_hygiene(hits)
    v = h[h["valid_hit"]]
    pool = v["drift_hz_s"].to_numpy(float)
    out = {}
    for name in beam_names:
        col = f"{name}:on_pair_line"
        if col not in v:
            continue
        on = v[v[col].astype(bool)]
        res = drift_null(pool, v.index.to_numpy(), on.index.to_numpy(),
                         on["drift_hz_s"].to_numpy(float),
                         on[f"{name}:drift_centre_hz_s"].to_numpy(float),
                         on[f"{name}:halfwidth_tight_hz_s"].to_numpy(float), k_sig,
                         rfi_on=on[rfi_col].to_numpy(bool) if rfi_col in on else None,
                         drift_range_hz_s=drift_range_hz_s)
        res["hits"] = [{"target": str(r["target"]), "freq_mhz": float(r["freq_mhz"]),
                        "drift_hz_s": float(r["drift_hz_s"]), "p_chance": p,
                        "drift_match": bool(abs(float(r["drift_hz_s"]) - float(r[f"{name}:drift_centre_hz_s"]))
                                            <= k_sig * float(r[f"{name}:halfwidth_tight_hz_s"]))}
                       for (_, r), p in zip(on.iterrows(), res.pop("per_hit_p"), strict=False)]
        out[name] = res
    return out


# ---------------------------------------------------------------------------
# offline recomputation from a run's committed files
# ---------------------------------------------------------------------------
def _parse_utc(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def consistency_checks(summary: dict, hits_rep: dict, xm: pd.DataFrame, beam_names) -> dict:
    """Do the committed files describe the same run?  Every check is recorded."""
    chk = {}
    ts, th = _parse_utc(summary.get("generated_utc")), _parse_utc(hits_rep.get("generated_utc"))
    chk["summary_generated_utc"] = summary.get("generated_utc")
    chk["hits_generated_utc"] = hits_rep.get("generated_utc")
    chk["summary_after_hits_within_10min"] = bool(ts and th and 0 <= (ts - th).total_seconds() <= 600)
    chk["n_hits_summary_eq_hits_json_eq_csv"] = (
        (summary.get("funnel") or {}).get("n_hits") == hits_rep.get("n_hits") == len(xm))
    chk["n_matched_summary_eq_csv"] = (
        (summary.get("funnel") or {}).get("n_hits_on_sample_stars") == int((xm["gaia_idx"] >= 0).sum()))
    per = {}
    for name in beam_names:
        pb = (summary.get("per_beam") or {}).get(name, {})
        col = f"{name}:on_pair_line"
        if col not in xm:
            continue
        per[name] = {"on_pair_line": pb.get("n_hits_on_pair_line") == int(xm[col].sum()),
                     "candidates": pb.get("n_candidates") == int(xm[f"{name}:candidate"].sum())}
    chk["per_beam_counts_match_csv"] = per
    chk["n_candidates_eq_sum_per_beam"] = (
        len(summary.get("candidates") or []) ==
        sum(int((summary.get("per_beam") or {}).get(n, {}).get("n_candidates") or 0) for n in beam_names))
    chk["all_consistent"] = bool(chk["summary_after_hits_within_10min"] and
                                 chk["n_hits_summary_eq_hits_json_eq_csv"] and
                                 chk["n_matched_summary_eq_csv"] and chk["n_candidates_eq_sum_per_beam"] and
                                 all(all(v.values()) for v in per.values()))
    return chk


def offline_chance(out_dir: Path, conf: dict, *, source_run: str | None = None) -> dict:
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    hits_rep = json.loads((out_dir / "hits.json").read_text())
    xm = pd.read_csv(out_dir / "hits_crossmatch.csv")
    beam_names = list((summary.get("per_beam") or {}).keys())
    k_sig = float(conf["assess"]["match_window_sigma"])
    aconf = conf["assess"]
    chk = consistency_checks(summary, hits_rep, xm, beam_names)

    h = hit_hygiene(xm)
    h["rfi_known_band"] = known_rfi_band(h["freq_mhz"], aconf.get("known_rfi_bands_mhz"))
    h["rfi_recurrent_wide"] = wide_recurrence(h, float(aconf.get("recurrence_wide_khz", 500.0)),
                                              int(aconf["recurrence_min_sightlines"]), valid=h["valid_hit"])
    h["rfi_any"] = h["rfi_flag"].astype(bool) | h["rfi_known_band"].notna() | h["rfi_recurrent_wide"]

    # (a) exactly as the run counted: every row, the run's own RFI flag
    raw = h.copy()
    raw["valid_hit"] = np.isfinite(raw["freq_mhz"]) & np.isfinite(raw["drift_hz_s"])
    as_run = assess_chance(raw, beam_names, k_sig=k_sig, rfi_col="rfi_flag")
    # (b) after hygiene, with the run's RFI flag
    cleaned = assess_chance(h, beam_names, k_sig=k_sig, rfi_col="rfi_flag")
    # (c) after hygiene, with the known-band and wide-recurrence rules added
    cleaned_rfi = assess_chance(h, beam_names, k_sig=k_sig, rfi_col="rfi_any")

    funnel = summary.get("funnel") or {}
    geom_null = {n: geometry_null((summary["per_beam"][n] or {}).get("n_bl_targets_as_transmitter"),
                                  funnel.get("n_bl_targets_in_sample")) for n in beam_names}
    cands = summary.get("candidates") or []
    uniq = {(str(c["target"]), float(c["freq_mhz"]), float(c["drift_hz_s"])) for c in cands}
    matched = h[h["gaia_idx"] >= 0]
    rep = {"stage": "chance_offline",
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source_run": source_run,
           "source_files": ["summary.json", "hits.json", "hits_crossmatch.csv"],
           "consistency": chk,
           "n_hit_beam_matches_reported": len(cands),
           "n_unique_hits_behind_them": len(uniq),
           "hygiene": {"n_rows": int(len(h)),
                       "n_not_a_frequency": int(h["artefact_not_frequency"].sum()),
                       "n_injection_rows": int(h["artefact_injection_table"].sum()),
                       "n_duplicate_rows": int(h["duplicate_row"].sum()),
                       "n_valid": int(h["valid_hit"].sum()),
                       "not_a_frequency_rows": h.loc[h["artefact_not_frequency"],
                                                     ["target", "freq_mhz", "drift_hz_s", "table", "freq_header"]
                                                     ].to_dict("records")},
           "rfi_on_sample_hits": matched[["target", "freq_mhz", "drift_hz_s", "table", "valid_hit",
                                          "rfi_flag", "rfi_known_band", "rfi_recurrent_wide"]
                                         ].to_dict("records"),
           "geometry_null_fraction_of_bl_targets_on_a_pair_line": geom_null,
           "match_window_sigma": k_sig,
           "chance_as_run": as_run,
           "chance_after_hygiene": cleaned,
           "chance_after_hygiene_and_band_rfi": cleaned_rfi}
    tot = lambda d, k: round(sum(v[k] for v in d.values()), 3)  # noqa: E731
    rep["totals"] = {
        "as_run": {"observed_candidates": tot(as_run, "n_candidates_after_rfi"),
                   "expected_candidates": tot(as_run, "n_expected_candidates")},
        "after_hygiene": {"observed_candidates": tot(cleaned, "n_candidates_after_rfi"),
                          "expected_candidates": tot(cleaned, "n_expected_candidates")},
        "after_hygiene_and_band_rfi": {"observed_candidates": tot(cleaned_rfi, "n_candidates_after_rfi"),
                                       "expected_candidates": tot(cleaned_rfi, "n_expected_candidates")}}
    rep["verdict"] = (f"CHANCE_LEVEL ({rep['totals']['as_run']['observed_candidates']:.0f} hit-beam "
                      f"matches vs {rep['totals']['as_run']['expected_candidates']:.1f} expected from the "
                      f"hits' own drift distribution; {rep['totals']['after_hygiene_and_band_rfi']['observed_candidates']:.0f}"
                      f" survive hygiene + band RFI)")
    return rep


def main(argv=None) -> int:
    from .run import load_relay_config

    p = argparse.ArgumentParser(prog="seti.relay.chance")
    p.add_argument("--out-dir", default="results/relay")
    p.add_argument("--source-run", default=None)
    p.add_argument("--annotate-summary", action="store_true",
                   help="also write the chance totals into summary.json as chance_offline")
    a = p.parse_args(argv)
    conf = load_relay_config()
    rep = offline_chance(Path(a.out_dir), conf, source_run=a.source_run)
    tag = f"_run{a.source_run}" if a.source_run else ""
    path = Path(a.out_dir) / f"chance{tag}.json"
    path.write_text(json.dumps(rep, indent=2, default=str) + "\n")
    if a.annotate_summary:
        sp = Path(a.out_dir) / "summary.json"
        s = json.loads(sp.read_text())
        s["chance_offline"] = {"file": path.name, "generated_utc": rep["generated_utc"],
                               "source_run": a.source_run, "verdict": rep["verdict"],
                               "consistency_all": rep["consistency"]["all_consistent"],
                               "totals": rep["totals"],
                               "n_unique_hits_behind_matches": rep["n_unique_hits_behind_them"],
                               "per_beam_expected_candidates_as_run":
                                   {k: v["n_expected_candidates"] for k, v in rep["chance_as_run"].items()},
                               "geometry_null": rep["geometry_null_fraction_of_bl_targets_on_a_pair_line"]}
        sp.write_text(json.dumps(s, indent=2, default=str) + "\n")
    print(f"[relay.chance] {rep['verdict']} -> {path}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())

"""Observed-frame triage of the committed laser-line candidate lists.

The in-funnel known-line rejection tests a candidate against the line list
shifted to the *catalogue* redshift/RV.  That is the right first cut, but it has
a leak: when the catalogue RV is wrong, imprecise, or belongs to the wrong blend
component, an ordinary stellar line (He I 5876, the Ca II triplet, O I 8446...)
lands a few Angstrom away from where the funnel looked and sails through as a
"candidate".  The committed candidate lists show exactly this signature: top
"survivors" sitting on famous stellar lines, and wavelength clusters recurring
across unrelated sightlines.

This module is the post-hoc, fully offline second pass that closes the leak:

* **duplicate**             --- the same (spec_id, wavelength) row appearing twice
                                (runs merged upstream);
* **known_line_window**     --- within a +-``v_window_kms`` velocity window of ANY
                                known stellar / ISM / DIB / sky / nebular line in
                                the observed (vacuum) frame.  A laser can sit
                                anywhere in the band, so excluding +-300 km/s
                                around every known line costs only a few percent
                                of bandwidth (reported in the summary) while
                                removing the entire catalogue-RV-error leak;
* **recurrent_across_runs** --- the same observed wavelength (+-``recur_tol`` A)
                                hosting candidates in >= ``recur_min`` distinct
                                spectra across ALL runs and BOTH modes.  A real
                                beacon has no reason to repeat at one instrumental
                                wavelength across unrelated sightlines; a CCD
                                defect, a bad sky model bin, or a common stellar
                                line does;
* **survives**              --- none of the above: the shortlist worth spending
                                confirmation-workflow time on.

Everything here runs on the committed ``results/*/laser_candidates.csv`` with no
network, so it is unit-testable and rerunnable in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .absorb import DIB_LINES, ISM_LINES, REST_ABSORPTION_LINES
from .reject import REST_EMISSION_LINES, SKY_LINES

C_KMS = 299792.458

# The observed-frame optical band the searches cover, for the bandwidth report.
BAND_MIN_A = 3600.0
BAND_MAX_A = 9800.0


def known_lines_vacuum() -> np.ndarray:
    """Union of every known-line list, on the vacuum scale, deduplicated."""
    lines = np.concatenate([
        REST_EMISSION_LINES, REST_ABSORPTION_LINES, ISM_LINES, DIB_LINES,
        SKY_LINES,
    ])
    return np.unique(np.round(lines, 2))


def nearest_line_dv_kms(wave_obs, lines) -> tuple[np.ndarray, np.ndarray]:
    """(|velocity offset| km/s, nearest line) for each observed wavelength."""
    w = np.atleast_1d(np.asarray(wave_obs, dtype=float))
    idx = np.searchsorted(lines, w)
    idx_lo = np.clip(idx - 1, 0, lines.size - 1)
    idx_hi = np.clip(idx, 0, lines.size - 1)
    d_lo = np.abs(w - lines[idx_lo])
    d_hi = np.abs(w - lines[idx_hi])
    nearest = np.where(d_lo <= d_hi, lines[idx_lo], lines[idx_hi])
    return np.abs(w - nearest) / nearest * C_KMS, nearest


def excluded_bandwidth_frac(lines: np.ndarray, v_window_kms: float,
                            lo: float = BAND_MIN_A, hi: float = BAND_MAX_A) -> float:
    """Fraction of [lo, hi] covered by the union of +-v windows around the lines."""
    half = lines * v_window_kms / C_KMS
    ivals = sorted((max(c - h, lo), min(c + h, hi))
                   for c, h in zip(lines, half, strict=True)
                   if c + h > lo and c - h < hi)
    covered, cur_lo, cur_hi = 0.0, None, None
    for a, b in ivals:
        if cur_hi is None or a > cur_hi:
            if cur_hi is not None:
                covered += cur_hi - cur_lo
            cur_lo, cur_hi = a, b
        else:
            cur_hi = max(cur_hi, b)
    if cur_hi is not None:
        covered += cur_hi - cur_lo
    return covered / (hi - lo)


def triage_candidates(cand: pd.DataFrame, v_window_kms: float = 300.0,
                      recur_tol: float = 3.0, recur_min: int = 3) -> pd.DataFrame:
    """Assign a ``triage_verdict`` to every candidate row.

    ``cand`` is the concatenation of all committed candidate lists (both modes,
    all runs) with at least ``spec_id`` and ``wavelength`` columns.  Recurrence is
    deliberately computed across the full concatenation: an instrumental
    wavelength repeats regardless of which run or mode found it.
    """
    out = cand.copy().reset_index(drop=True)
    wave = pd.to_numeric(out["wavelength"], errors="coerce").to_numpy()

    dup = out.assign(_w=np.round(wave, 1)).duplicated(subset=["spec_id", "_w"])

    lines = known_lines_vacuum()
    dv, nearest = nearest_line_dv_kms(wave, lines)
    near_known = dv <= v_window_kms

    # Cross-run recurrence: count DISTINCT spectra with a candidate within
    # +-recur_tol of each row's wavelength (self included).
    order = np.argsort(wave)
    w_sorted = wave[order]
    ids_sorted = out["spec_id"].astype(str).to_numpy()[order]
    n_recur = np.zeros(len(out), dtype=int)
    lo_idx = np.searchsorted(w_sorted, wave - recur_tol, side="left")
    hi_idx = np.searchsorted(w_sorted, wave + recur_tol, side="right")
    for i, (a, b) in enumerate(zip(lo_idx, hi_idx, strict=True)):
        n_recur[i] = len(set(ids_sorted[a:b]))
    recurrent = n_recur >= recur_min

    verdict = np.where(dup, "duplicate",
               np.where(near_known, "known_line_window",
               np.where(recurrent, "recurrent_across_runs", "survives")))
    out = out.assign(nearest_known_line=np.round(nearest, 2),
                     nearest_line_dv_kms=np.round(dv, 1),
                     n_spectra_at_wavelength=n_recur,
                     triage_verdict=verdict).drop(columns=["_w"], errors="ignore")

    # Final stage: a candidate whose surviving lines in the *same* spectrum form a
    # nebular emission family at one redshift is a misclassified emission-line
    # galaxy, not a transmitter (this unmasked the former #1 target, Halpha+[N II]
    # at z=0.145).  Test the survivors only and override their verdict.
    from .galaxy_reject import flag_galaxy_spectra
    out["galaxy_z"] = np.nan
    out["galaxy_rest_lines"] = ""
    surv_mask = out["triage_verdict"] == "survives"
    if surv_mask.any():
        gflag = flag_galaxy_spectra(out[surv_mask])
        gal = gflag[gflag["is_galaxy"]]
        if len(gal):
            gz = gal.groupby("spec_id")[["galaxy_z", "galaxy_rest_lines"]].first()
            is_gal = out["spec_id"].isin(gz.index) & surv_mask
            out.loc[is_gal, "triage_verdict"] = "galaxy_zmatch"
            out.loc[is_gal, "galaxy_z"] = out.loc[is_gal, "spec_id"].map(
                gz["galaxy_z"])
            out.loc[is_gal, "galaxy_rest_lines"] = out.loc[is_gal, "spec_id"].map(
                gz["galaxy_rest_lines"])
    return out


def triage_run(root: Path | str, v_window_kms: float = 300.0,
               recur_tol: float = 3.0, recur_min: int = 3) -> dict:
    """Triage every committed candidate list; write the shortlist + summary."""
    root = Path(root)
    frames = []
    for sub, mode in (("spectra", "emission"), ("spectra_absorption", "absorption")):
        path = root / "results" / sub / "laser_candidates.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df.assign(search_mode=mode))
    if not frames:
        raise FileNotFoundError("no committed laser_candidates.csv found")
    cand = pd.concat(frames, ignore_index=True)

    triaged = triage_candidates(cand, v_window_kms=v_window_kms,
                                recur_tol=recur_tol, recur_min=recur_min)
    survivors = (triaged[triaged["triage_verdict"] == "survives"]
                 .sort_values("significance", ascending=False))

    out_dir = root / "results" / "spectra_triage"
    out_dir.mkdir(parents=True, exist_ok=True)
    triaged.to_csv(out_dir / "triaged_candidates.csv", index=False)
    survivors.to_csv(out_dir / "priority_targets.csv", index=False)

    counts = triaged["triage_verdict"].value_counts().to_dict()
    summary = {
        "n_input": int(len(triaged)),
        "verdict_counts": {k: int(v) for k, v in counts.items()},
        "n_priority_targets": int(len(survivors)),
        "v_window_kms": v_window_kms,
        "recur_tol_A": recur_tol,
        "recur_min_spectra": recur_min,
        "excluded_bandwidth_frac": round(
            excluded_bandwidth_frac(known_lines_vacuum(), v_window_kms), 4),
        "per_mode": {m: int(n) for m, n in
                     survivors["search_mode"].value_counts().items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[triage]", json.dumps(summary))
    return summary


__all__ = ["triage_candidates", "triage_run", "known_lines_vacuum",
           "nearest_line_dv_kms", "excluded_bandwidth_frac"]

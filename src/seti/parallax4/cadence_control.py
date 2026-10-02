"""Fixed-cadence synthetic phasing controls; never an empirical sky calibration.

Read-only diagnostic. Sampling blocks are adjacent observing opportunities,
not recovered dips or independent physical episodes. Production classification,
fitted ephemerides, thresholds, preregistrations and historical outputs are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from astropy.io.votable import parse_single_table

from . import deepvet as DV
from . import epochs as EP
from .greydip import GreyConfig

FIXTURE_PATH = "tests/fixtures/parallax4/dr3_epoch_photometry_1457486023639239296.vot"
FIXTURE_GIT_BLOB = "06eddfdeeb713769aaa78d08843ef23b3fe02857"
FIXTURE_SHA256 = "664c9f20d750c552c4eb1230db537b1025c7dd7e60fc6ba56efadfff9b35544f"
SOURCE_ID = "1457486023639239296"
INPUT_COMMIT = "700be390a9cdb9d9af68e0679cb128b3a212167b"
README_GIT_BLOB = "7050b9b33d914fd73f857e8ffd4a4a0f94ac02fd"


def _times(values) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("times must be a nonempty finite one-dimensional vector")
    return times


def sampling_blocks(times, gap_d: float = 1.0) -> list[list[float]]:
    """Unique epochs, linked by adjacent gap <= gap_d; starts are observed epochs.

    These are sampling proxies only: the real dip extractor can break a chain
    at a normal usable transit, and has additional flux/sign criteria.
    """
    if not np.isfinite(gap_d) or gap_d < 0:
        raise ValueError("gap_d must be finite and nonnegative")
    ordered = np.unique(_times(times))
    groups: list[list[float]] = []
    for time in ordered:
        if not groups or time - groups[-1][-1] > gap_d:
            groups.append([])
        groups[-1].append(float(time))
    return groups


def alias_masks(times, period: float, duration: float, tol: float,
                t0: float = 0.0) -> tuple[list[dict], np.ndarray]:
    """Production circular acceptance, conditioned on an externally fixed ephemeris."""
    times = _times(times)
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite")
    _, trials = DV._ztf_phase_chance_bound(period, duration, tol, len(times))
    if not all(np.isfinite(x["period_d"]) and x["period_d"] > 0 for x in trials):
        raise ValueError("derived alias periods must be finite and positive")
    if not np.isfinite(times - t0).all():
        raise ValueError("epoch differences must be finite")
    masks = []
    for trial in trials:
        ph = ((times - t0) / trial["period_d"] + 0.5) % 1.0 - 0.5
        half = trial["half_width_phase"]
        masks.append((np.abs(ph) <= half) | (np.abs(np.abs(ph) - 0.5) <= half))
    return trials, np.asarray(masks)


def subset_null(times, k: int, period: float, duration: float, tol: float,
                t0: float = 0.0) -> dict:
    """Exact alias union for uniform k-subset labels WITHOUT replacement.

    Exchangeability is a synthetic assumption, not evidence about actual dips.
    Inclusion-exclusion over the three alias acceptance sets needs no
    independence between aliases.
    """
    times = _times(times)
    if len(np.unique(times)) != len(times):
        raise ValueError("subset opportunities must be unique")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) \
            or not 1 <= k <= len(times):
        raise ValueError("k must be an integer in [1, number of opportunities]")
    trials, masks = alias_masks(times, period, duration, tol, t0)
    numerator = 0
    terms = []
    for order in (1, 2, 3):
        for indices in combinations(range(3), order):
            count = int(np.all(masks[list(indices)], axis=0).sum())
            subsets = math.comb(count, int(k)) if count >= k else 0
            sign = 1 if order % 2 else -1
            numerator += sign * subsets
            terms.append({"alias_indices": list(indices), "intersection_size": count,
                          "signed_subset_count": str(sign * subsets)})
    denominator = math.comb(len(times), int(k))
    return {"model": "UNIFORM_K_SUBSET_WITHOUT_REPLACEMENT",
            "opportunity_count": len(times), "label_count": int(k),
            "alias_periods_d": [x["period_d"] for x in trials],
            "accepted_counts": masks.sum(axis=1).astype(int).tolist(),
            "inclusion_exclusion_terms": terms, "union_subset_count": str(numerator),
            "all_subset_count": str(denominator), "probability": numerator / denominator}


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _intersect(left, right) -> list[tuple[float, float]]:
    return _merge([(max(a, c), min(b, d)) for a, b in left for c, d in right
                   if min(b, d) > max(a, c)])


def common_shift_null(times, period: float, duration: float, tol: float,
                      t0: float = 0.0) -> dict:
    """One shared shift delta ~ Uniform[0, 2P), preserving all relative times.

    Exact interval integration under fixed real-valued input times (float64
    arithmetic). This is not independent per-slot phase randomization.
    Duplicate times remain correlated slots; unique_epoch_count exposes this.
    """
    times = _times(times)
    trials, _ = alias_masks(times, period, duration, tol, t0)
    length = 2 * period  # common period of every alias acceptance pattern
    per_alias = []
    union_intervals: list[tuple[float, float]] = []
    for trial in trials:
        p = trial["period_d"]
        spacing = p / 2  # primary and secondary centers
        half_d = duration / 2 + tol * p
        accepted = [(0.0, length)]
        if half_d < spacing / 2:
            for time in times:
                center = (t0 - float(time)) % spacing
                windows = _merge([
                    (max(0.0, center + j * spacing - half_d),
                     min(length, center + j * spacing + half_d))
                    for j in range(-1, round(length / spacing) + 1)
                ])
                accepted = _intersect(accepted, windows)
                if not accepted:
                    break
        measure = sum(hi - lo for lo, hi in accepted)
        per_alias.append({"period_d": p, "accepted_shift_intervals_d": accepted,
                          "probability": measure / length})
        union_intervals.extend(accepted)
    union_intervals = _merge(union_intervals)
    return {"model": "ONE_COMMON_ORIGIN_SHIFT",
            "shift_support_d": [0.0, length], "slot_count": len(times),
            "unique_epoch_count": len(np.unique(times)), "alias_trials": per_alias,
            "union_shift_intervals_d": union_intervals,
            "probability": sum(hi - lo for lo, hi in union_intervals) / length}


def load_fixture(path: Path) -> tuple[dict, list[dict]]:
    """Refuse altered bytes/identity/time metadata before applying the G quality mask."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("PINNED_FIXTURE_SHA256_MISMATCH")
    root = ET.fromstring(raw)
    ns = {"v": "http://www.ivoa.net/xml/VOTable/v1.3"}
    ids = [x.get("value") for x in root.findall(".//v:PARAM", ns)
           if x.get("name") == "source_id"]
    frames = root.findall(".//v:TIMESYS", ns)
    if ids != [SOURCE_ID] or len(frames) != 1:
        raise ValueError("PINNED_FIXTURE_IDENTITY_OR_TIME_FRAME_MISMATCH")
    frame = frames[0]
    expected_frame = {"refposition": "BARYCENTER", "timescale": "TCB",
                      "timeorigin": "2455197.5"}
    if any(frame.get(key) != value for key, value in expected_frame.items()):
        raise ValueError("PINNED_FIXTURE_TIME_FRAME_MISMATCH")
    table = parse_single_table(path).to_table()
    phot = EP.photometry_from_long(table, source_id=int(SOURCE_ID))
    cfg = GreyConfig()
    mask = (np.isfinite(phot["t"]) & ~phot["bad_g"]
            & np.isfinite(phot["f_g"]) & np.isfinite(phot["e_g"])
            & (phot["f_g"] > 0) & (phot["e_g"] > 0)
            & (phot["e_g"] < cfg.max_band_err_frac * np.abs(phot["f_g"])))
    usable = phot[mask].sort_values("t")
    if usable.empty:
        raise ValueError("NO_ELIGIBLE_G_EPOCHS")
    records = [{"t_gaia_tcb_d": float(row.t), "transit_id": str(int(row.transit_id)),
                "vrej_g_diagnostic": bool(row.vrej_g)}
               for row in usable.itertuples()]
    meta = {"source_id": SOURCE_ID, "fixture_path": FIXTURE_PATH,
            "fixture_git_blob": FIXTURE_GIT_BLOB, "fixture_sha256": FIXTURE_SHA256,
            "fixture_bytes": len(raw), "input_commit": INPUT_COMMIT,
            "provenance_readme_git_blob": README_GIT_BLOB,
            "acquisition": "EXISTING_2026_09_22_PUBLIC_REDISTRIBUTION_NOT_NEW_ESA_FETCH",
            "time_frame": expected_frame, "time_unit": "day",
            "g_error_fraction_cut_strict": cfg.max_band_err_frac,
            "raw_rows": len(phot), "eligible_rows": len(usable),
            "eligible_unique_epochs": int(usable["t"].nunique()),
            "vrej_g_eligible_diagnostic_count": int(usable["vrej_g"].sum()),
            "bad_g_count": int(phot["bad_g"].sum())}
    return meta, records


def build_report(path: Path) -> dict:
    meta, records = load_fixture(path)
    cfg = GreyConfig()
    groups = sampling_blocks([x["t_gaia_tcb_d"] for x in records], cfg.episode_gap_d)
    starts = [x[0] for x in groups]
    period, duration, tol, t0, k = 1.0, 0.1, 0.03, 0.0, 10
    _, masks = alias_masks(starts, period, duration, tol, t0)
    matching = [t for t, ok in zip(starts, masks[2]) if ok]
    if len(starts) < k or len(matching) < k:
        raise ValueError("INSUFFICIENT_PREDECLARED_CONTROL_OPPORTUNITIES")
    chronological = starts[:k]
    positive = matching[:k]
    iid_bound, iid_trials = DV._ztf_phase_chance_bound(period, duration, tol, k)
    return {
        "schema_version": 1, "ai_authorship": "OpenAI Codex; independent review recorded in handoff",
        "result_kind": "CONDITIONAL_SYNTHETIC_CONTROL",
        "source": meta, "eligible_observations": records,
        "sampling_proxy": {"rule": "ADJACENT_GAP_LE_1_DAY_UNIQUE_EPOCHS",
                           "gap_d": cfg.episode_gap_d,
                           "representative": "FIRST_OBSERVED_EPOCH",
                           "not_recovered_dip_episodes": True,
                           "block_sizes": [len(g) for g in groups], "block_starts_d": starts},
        "fixed_ephemeris": {"period_d": period, "duration_d": duration,
                            "t0_gaia_tcb_d": t0, "phase_tol": tol,
                            "label_count": k, "selection": "PREDECLARED_SYNTHETIC_NOT_FLUX_FITTED"},
        "positive_control": {"selection": "FIRST_10_CHRONOLOGICAL_P_HALF_MATCHING_BLOCK_STARTS",
                             "available_matching_count": len(matching), "times_d": positive,
                             "purpose": "PHASE_PLUMBING_ONLY_NOT_BLS_OR_PHOTOMETRIC_RECOVERY"},
        "uniform_subset_null": subset_null(starts, k, period, duration, tol, t0),
        "chronological_common_shift_null": {
            "selection": "FIRST_10_CHRONOLOGICAL_BLOCK_STARTS_INDEPENDENT_OF_PHASE",
            "times_d": chronological,
            **common_shift_null(chronological, period, duration, tol, t0)},
        "phase_selected_positive_shift_sensitivity": {
            "selection": "SAME_PHASE_SELECTED_POSITIVE_NOT_MAIN_NULL",
            **common_shift_null(positive, period, duration, tol, t0)},
        "perfect_correlation_toy": {
            "selection": "ONE_SYNTHETIC_EPOCH_REPEATED_10_SLOTS_NOT_10_EPISODES",
            **common_shift_null([0.0] * k, period, duration, tol, t0)},
        "iid_uniform_comparison": {"assumption": "INDEPENDENT_UNIFORM_EPISODE_PHASES",
                                   "union_bound": iid_bound, "alias_trials": iid_trials},
        "limitations": [
            "Fixed observed cadence and synthetic exchangeable subset labels only.",
            "Common shift is one shared random variable; chronology and relative cadence persist.",
            "Sampling blocks are not recovered peaks or independent physical episodes.",
            "Five variability-rejection flags are diagnostic, not production-mask exclusions.",
            "Positive is phase-selected and tests plumbing, not recovery or independent validation.",
            "Ephemeris is predeclared; fitted-ephemeris uncertainty and search selection are unmodeled.",
            "No empirical sky false-positive rate, calibration, detection or threshold change.",
            "Existing redistributed fixture; no new direct ESA/catalogue acquisition."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(FIXTURE_PATH))
    args = parser.parse_args()
    report = build_report(args.fixture)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    print("CADENCE_CONTROL_JSON " + canonical.rstrip("\n"))
    print("CADENCE_CONTROL_REPORT_SHA256 " + hashlib.sha256(canonical.encode()).hexdigest())


if __name__ == "__main__":
    main()

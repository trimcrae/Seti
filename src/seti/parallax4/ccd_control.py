"""One conditional matched-mean CCD discordance control; no science outputs written."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import epochs as E
from . import photocentre as P
from . import spatial_control as S

ENTRY_COMMIT = "b9b2f9ef55b4737396052007de58e89ea541af8d"
SYNTHETIC_D_MAS = (300.0, -200.0)
PSI_AMPLITUDE = 0.2


def eligible_ccds(ccd):
    """The existing collapse kernel's mask, with no inferred flag-bit meanings."""
    return (np.isfinite(ccd["x_al"]) & np.isfinite(ccd["sx_al"]) & (ccd["sx_al"] > 0)
            & np.isfinite(ccd["pf_al"]) & np.isfinite(ccd["theta"]) & ccd["bary_ok"]
            & ccd["used"])


def matched_centroids(ccd, offsets):
    """Reference/coherent and single-CCD arms with one prescribed transit mean.

    Only eligible x_al is replaced. The first eligible native array slot is
    selected independently of frame order. For W=sum(w) and selected w_j,
    its discordant centroid is delta*W/w_j, giving mean delta and
    chi2_ccd=delta**2*(W**2/w_j-W). No noise realisation is added.
    """
    reference, discordant = ccd.copy(), ccd.copy()
    eligible = ccd[eligible_ccds(ccd)]
    diagnostics = []
    for (sid, tid), group in eligible.groupby(["source_id", "transit_id"], sort=False):
        key = (int(sid), int(tid))
        if key not in offsets:
            raise ValueError("MISSING_SYNTHETIC_TRANSIT_OFFSET")
        group = group.sort_values("ccd_slot")
        if group["ccd_slot"].duplicated().any():
            raise ValueError("AMBIGUOUS_NATIVE_CCD_SLOT")
        delta = float(offsets[key])
        if not np.isfinite(delta):
            raise ValueError("NONFINITE_SYNTHETIC_TRANSIT_OFFSET")
        weights = 1.0 / group["sx_al"].to_numpy(float) ** 2
        total, selected = float(weights.sum()), float(weights[0])
        first = group.index[0]
        reference.loc[group.index, "x_al"] = delta
        discordant.loc[group.index, "x_al"] = 0.0
        discordant.loc[first, "x_al"] = delta * total / selected
        diagnostics.append({
            "source_id": str(int(sid)), "transit_id": str(int(tid)),
            "eligible_ccds": len(group), "selected_native_slot": int(group.iloc[0]["ccd_slot"]),
            "synthetic_transit_offset_mas": delta, "sum_weight_mas_minus2": total,
            "selected_weight_fraction": selected / total,
            "injected_discordant_centroid_mas": delta * total / selected,
            "analytic_formal_error_mas": 1.0 / np.sqrt(total),
            "analytic_discordant_chi2_ccd": delta ** 2 * total * (total / selected - 1.0),
        })
    return reference, discordant, diagnostics


def load_native_bindings():
    """Read only existing pinned fixtures; preserve native slot/error/use/flags."""
    metadata, joint = S.load_geometry()
    ccd = E.read_prerelease_zip(S.ZIP_PATH)
    required = {
        "source_id": "source_id", "transit_id": "transit_id",
        "centroid_al": "centroid_pos_al", "centroid_al_error": "centroid_pos_error_al",
        "used_al": "used_by_agis_al", "ccd_proc_flags": "ccd_proc_flags",
    }
    resolution = ccd.attrs["resolution"]["mapping"]
    if any(resolution.get(key) != value for key, value in required.items()):
        raise ValueError("NATIVE_CCD_BINDING_MISMATCH")
    native = ccd[ccd["source_id"] == int(S.SOURCE_ID)].copy()
    native["ccd_slot"] = native.groupby(["source_id", "transit_id"], sort=False).cumcount()
    sizes = native.groupby(["source_id", "transit_id"], sort=False).size()
    if not (sizes == 10).all():
        raise ValueError("NATIVE_CCD_ARRAY_LENGTH_MISMATCH")
    native = native[native["transit_id"].isin(joint["transit_id"])].copy()
    collapsed = E.collapse_transits(native)
    if set(collapsed["transit_id"]) != set(joint["transit_id"]):
        raise ValueError("NATIVE_CCD_TRANSIT_BINDING_MISMATCH")
    metadata = dict(metadata)
    metadata.update(entry_commit=ENTRY_COMMIT, exact_native_roles=required,
                    retained_native_ccd_slots=len(native),
                    retained_eligible_ccds=int(eligible_ccds(native).sum()),
                    processing_flag_bits_interpreted=False)
    return metadata, native, joint


def _fit_arm(ccd, joint, phi):
    collapsed = E.collapse_transits(ccd)
    geometry = joint[["source_id", "transit_id"]].copy()
    geometry["phi"] = phi
    geometry["sphi"] = 0.001
    fit_input = geometry.merge(collapsed, on=["source_id", "transit_id"],
                               validate="one_to_one", sort=False)
    if len(fit_input) != len(joint):
        raise ValueError("SYNTHETIC_CCD_JOIN_COVERAGE_CHANGED")
    return fit_input, P.fit_photocentre(fit_input)


def build_report():
    metadata, native, joint = load_native_bindings()
    index = np.arange(len(joint), dtype=float)
    psi_input = PSI_AMPLITUDE * np.sin(2 * np.pi * index * ((np.sqrt(5) - 1) / 2))
    phi = psi_input / (1 - psi_input)
    median = float(np.median(phi))
    psi_final = (1 + median) * psi_input - median
    offsets_values = psi_final * (SYNTHETIC_D_MAS[0] * np.sin(joint["theta"])
                                 + SYNTHETIC_D_MAS[1] * np.cos(joint["theta"]))
    offsets = {(int(row.source_id), int(row.transit_id)): float(delta)
               for row, delta in zip(joint.itertuples(), offsets_values, strict=True)}
    reference, discordant, analytic = matched_centroids(native, offsets)
    ref_input, ref_fit = _fit_arm(reference, joint, phi)
    dis_input, dis_fit = _fit_arm(discordant, joint, phi)
    by_id = {row["transit_id"]: row for row in analytic}
    rows = []
    for ref, dis in zip(ref_input.itertuples(), dis_input.itertuples(), strict=True):
        if ref.transit_id != dis.transit_id:
            raise ValueError("SYNTHETIC_TRANSIT_ORDER_MISMATCH")
        item = dict(by_id[str(int(ref.transit_id))])
        selected = native[native["transit_id"] == ref.transit_id].sort_values("ccd_slot")
        item.update(
            native_ccd_slots=selected["ccd_slot"].astype(int).tolist(),
            native_centroid_al_mas=selected["x_al"].to_numpy(float).tolist(),
            native_centroid_error_al_mas=selected["sx_al"].to_numpy(float).tolist(),
            native_used_by_agis_al=selected["used"].astype(bool).tolist(),
            native_processing_flags_uint16=selected["ccd_flags"].astype(int).tolist(),
            native_eligible_mask=eligible_ccds(selected).astype(bool).tolist(),
            native_theta_rad=selected["theta"].to_numpy(float).tolist(),
            native_time_gaia_d=selected["t"].to_numpy(float).tolist(),
            reference_x_al_mas=float(ref.x_al), discordant_x_al_mas=float(dis.x_al),
            reference_sx_al_mas=float(ref.sx_al), discordant_sx_al_mas=float(dis.sx_al),
            reference_chi2_ccd=float(ref.chi2_ccd), discordant_chi2_ccd=float(dis.chi2_ccd),
        )
        rows.append(item)
    return {
        "schema_version": 1,
        "result_kind": "CONDITIONAL_SYNTHETIC_MATCHED_MEAN_CCD_DISCORDANCE_CONTROL",
        "ai_authorship": "OpenAI Codex AI-assisted; independently reviewed",
        "source": metadata,
        "predeclared_control": {
            "synthetic_D_mas": list(SYNTHETIC_D_MAS), "psi_amplitude": PSI_AMPLITUDE,
            "flux_sequence": "SIN_2PI_CHRONOLOGICAL_INDEX_GOLDEN_RATIO_FRACTION",
            "reference": "SAME_SYNTHETIC_AL_CENTROID_ON_EVERY_ELIGIBLE_CCD",
            "discordant": "FIRST_ELIGIBLE_NATIVE_SLOT_DELTA_TIMES_SUM_WEIGHT_OVER_SELECTED_WEIGHT",
            "native_geometry_errors_use_and_flags": "UNCHANGED",
            "slot_identity": "ORIGINAL_ARRAY_ORDINAL_NOT_DOCUMENTED_PHYSICAL_CCD_ID",
            "native_centroids": "RECORDED_SEPARATELY; ELIGIBLE_CENTROIDS_REPLACED",
            "noise_realisation": "NONE; DETERMINISTIC_CONDITIONAL_COMPARISON",
        },
        "native_bound_transit_rows": rows,
        "reference_fit": ref_fit, "discordant_fit": dis_fit,
        "comparison": {
            "max_abs_transit_mean_difference_mas": float(np.max(np.abs(
                ref_input["x_al"].to_numpy() - dis_input["x_al"].to_numpy()))),
            "max_abs_formal_error_difference_mas": float(np.max(np.abs(
                ref_input["sx_al"].to_numpy() - dis_input["sx_al"].to_numpy()))),
            "reference_chi2_ccd_max": float(ref_input["chi2_ccd"].max()),
            "discordant_chi2_ccd_min": float(dis_input["chi2_ccd"].min()),
            "discordant_chi2_ccd_max": float(dis_input["chi2_ccd"].max()),
            "diagnostic": "FIT_HAS_NO_CCD_COHERENCE_DISCRIMINANT_AFTER_MATCHED_MEAN_COLLAPSE",
        },
        "limitations": [
            "A conditional counterexample to a CCD-coherence interpretation of the transit fit.",
            "Synthetic flux and eligible centroids on native geometry, not measured contamination.",
            "No contamination rate, detection, sky false-positive rate or A4 calibration claim.",
            "Formal errors are unchanged; no new covariance model or production repair is justified.",
            "CCD flags are retained as diagnostics; undocumented bit meanings are not inferred.",
            "The fixed AGIS mask conditions on published eligibility, not acceptance of injected corruption.",
            "Exact algebra assumes the specified synthetic per-transit AL offsets and no added noise.",
            "Concentrated centroid amplitudes are a controlled construction, not a realistic corruption-frequency model.",
            "Matched transit means make residual jitter unable to distinguish these two arms.",
            "Scientific thresholds, release gates, native fixtures and historical results unchanged.",
        ],
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    report = _json_ready(build_report())
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print("CCD_CONTROL_JSON=" + payload)
    print("CCD_CONTROL_SHA256=" + hashlib.sha256((payload + "\n").encode()).hexdigest())


if __name__ == "__main__":
    main()

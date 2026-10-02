"""Read-only conditional spatial-identifiability control on pinned native geometry."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from astropy.table import Table

from . import epochs as E
from . import photocentre as P

SOURCE_ID = "1457486023639239296"
BASE_COMMIT = "b88e575c8d3d2218b7c1b6bea72003f74063fd3d"
ZIP_PATH = Path("tests/fixtures/parallax4/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
ZIP_BLOB = "b3d51a1506ca1d20f304a0895c294b4a9908039f"
ZIP_SHA256 = "07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b"
PHOT_PATH = Path("tests/fixtures/parallax4/dr3_epoch_photometry_1457486023639239296.vot")
PHOT_BLOB = "06eddfdeeb713769aaa78d08843ef23b3fe02857"
PHOT_SHA256 = "664c9f20d750c552c4eb1230db537b1025c7dd7e60fc6ba56efadfff9b35544f"
DRIFT_A_PER_YR = 0.1
HIDDEN_D_MAS = [300.0, -200.0]


def load_geometry() -> tuple[dict, object]:
    for path, expected in ((ZIP_PATH, ZIP_SHA256), (PHOT_PATH, PHOT_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("PINNED_SPATIAL_FIXTURE_SHA256_MISMATCH")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(
            (".xml", ".vot", ".votable"))]
        if len(names) != 1:
            raise ValueError("AMBIGUOUS_NATIVE_ASTROMETRY_MEMBER")
        raw_xml = archive.read(names[0])
    xml = ET.fromstring(raw_xml)
    frames = [dict(node.attrib) for node in xml.iter()
              if node.tag.rsplit("}", 1)[-1] == "TIMESYS"]
    ccd = E.read_prerelease_zip(ZIP_PATH)
    native = ccd[ccd["source_id"] == int(SOURCE_ID)]
    phot = E.photometry_from_long(Table.read(PHOT_PATH, format="votable"),
                                  source_id=int(SOURCE_ID))
    collapsed = E.collapse_transits(native)
    joined = E.join_photometry_astrometry(phot, collapsed)
    joint = P.prepare_joint(joined)
    finite = np.ones(len(joint), dtype=bool)
    for key in ("t_yr", "theta", "pf_al", "x_al", "sx_al", "phi"):
        finite &= np.isfinite(joint[key])
    joint = joint[finite & (joint["sx_al"] > 0)].copy()
    if len(joint) < P.PhotocentreConfig().n_min or joint["transit_id"].duplicated().any():
        raise ValueError("MISSING_OR_AMBIGUOUS_NATIVE_JOINT_GEOMETRY")
    if set(joint["source_id"]) != {int(SOURCE_ID)}:
        raise ValueError("NATIVE_SOURCE_ID_MISMATCH")
    metadata = {
        "base_commit": BASE_COMMIT, "source_id": SOURCE_ID,
        "zip_path": str(ZIP_PATH), "zip_git_blob": ZIP_BLOB, "zip_sha256": ZIP_SHA256,
        "zip_bytes": ZIP_PATH.stat().st_size, "archive_member": names[0],
        "archive_member_sha256": hashlib.sha256(raw_xml).hexdigest(),
        "native_timesys_metadata": frames,
        "phot_path": str(PHOT_PATH), "phot_git_blob": PHOT_BLOB,
        "phot_sha256": PHOT_SHA256, "phot_bytes": PHOT_PATH.stat().st_size,
        "astrometry_release": ccd.attrs.get("release"),
        "existing_redistribution_not_new_direct_esa_fetch": True,
        "raw_phot_transits": len(phot),
        "native_transit_id_intersection": len(set(phot["transit_id"]) & set(native["transit_id"])),
        "collapsed_usable_astrometry_transits": len(collapsed),
        "retained_joint_transits": len(joint),
        "max_abs_join_time_difference_s": float(np.max(np.abs(
            joint["t"] - joint["t_astro"])) * 86400),
        "theta_conversion": "NATIVE_DEGREES_TO_RADIANS_THEN_CIRCULAR_TRANSIT_MEAN",
        "year_reference_gaia_days": E.J2017P5_DAYS, "days_per_year": 365.25,
    }
    return metadata, joint


def drift_case(joint):
    """Predeclared affine psi, whose sky regressors lie in nuisance span."""
    synthetic = joint.copy()
    t = synthetic["t_yr"].to_numpy(float)
    t_ref = float(np.median(t))
    psi_input = DRIFT_A_PER_YR * (t - t_ref)
    if not np.isfinite(psi_input).all() or np.any(1 - psi_input <= 0):
        raise ValueError("PREDECLARED_SYNTHETIC_FLUX_NOT_POSITIVE")
    phi_input = psi_input / (1 - psi_input)
    median_phi = float(np.median(phi_input))
    # Exact algebra of the production median re-reference, valid also for even n.
    a_final = (1 + median_phi) * DRIFT_A_PER_YR
    b_final = -a_final * t_ref - median_phi
    psi_final = a_final * t + b_final
    synthetic["phi"] = phi_input
    synthetic["sphi"] = 0.001  # declared synthetic flux error, not the native error
    s, c = np.sin(synthetic["theta"]), np.cos(synthetic["theta"])
    design = np.column_stack([s, c, synthetic["pf_al"], t * s, t * c,
                              psi_final * s, psi_final * c, psi_final])
    nulls = np.zeros((2, 8))
    nulls[0, [0, 3, 5]] = [-b_final, -a_final, 1]
    nulls[1, [1, 4, 6]] = [-b_final, -a_final, 1]
    hidden = HIDDEN_D_MAS[0] * nulls[0] + HIDDEN_D_MAS[1] * nulls[1]
    injected_sky = psi_final * (HIDDEN_D_MAS[0] * s + HIDDEN_D_MAS[1] * c)
    cancelled = synthetic.copy()
    cancelled["x_al"] = 0.0
    sky_only = synthetic.copy()
    sky_only["x_al"] = injected_sky
    return {
        "synthetic_drift_a_per_yr": DRIFT_A_PER_YR, "t_ref_yr": t_ref,
        "hidden_D_mas": HIDDEN_D_MAS, "synthetic_sphi": 0.001,
        "median_phi_input": median_phi, "final_psi_a_per_yr": a_final,
        "final_psi_b": b_final, "min_total_flux_ratio": float(np.min(1 + phi_input)),
        "max_total_flux_ratio": float(np.max(1 + phi_input)),
        "analytic_target_null_vectors": nulls.tolist(),
        "max_abs_analytic_null_row_residual": float(np.max(np.abs(design @ nulls.T))),
        "max_abs_hidden_parameterization_difference_mas": float(
            np.max(np.abs(design @ hidden))),
        "synthetic_whitened_design_rank": int(np.linalg.matrix_rank(
            design / synthetic["sx_al"].to_numpy(float)[:, None])),
        "synthetic_whitened_design_columns": 8,
        "cancelled_hidden_blend_result": P.fit_photocentre(cancelled),
        "sky_only_injection_result": P.fit_photocentre(sky_only),
    }


def independent_flux_cases(joint):
    """Predeclared non-affine and near-affine controls; no verdict tuning."""
    j = np.arange(len(joint), dtype=float)
    modulation = np.sin(2 * np.pi * j * ((np.sqrt(5) - 1) / 2))
    t = joint["t_yr"].to_numpy(float)
    affine = DRIFT_A_PER_YR * (t - np.median(t))
    cases = {}
    for name, psi_input, zero_pf, zero_centroid in [
        ("nondegenerate_sky_positive", 0.2 * modulation, False, False),
        ("nuisance_only_parallax_null_positive", 0.2 * modulation, True, False),
        ("near_affine_zero_centroid", affine + 1e-6 * modulation, False, True),
    ]:
        synthetic = joint.copy()
        if np.any(1 - psi_input <= 0):
            raise ValueError("PREDECLARED_SYNTHETIC_FLUX_NOT_POSITIVE")
        phi_input = psi_input / (1 - psi_input)
        M = float(np.median(phi_input))
        psi_final = (1 + M) * psi_input - M
        synthetic["phi"], synthetic["sphi"] = phi_input, 0.001
        if zero_pf:
            synthetic["pf_al"] = 0.0  # explicitly synthetic nuisance-only column loss
        synthetic["x_al"] = (0.0 if zero_centroid else psi_final * (
            HIDDEN_D_MAS[0] * np.sin(synthetic["theta"])
            + HIDDEN_D_MAS[1] * np.cos(synthetic["theta"])))
        cases[name] = {"synthetic_changes": {
            "flux_sequence": "SIN_2PI_CHRONOLOGICAL_INDEX_GOLDEN_RATIO_FRACTION",
            "psi_modulation_amplitude": 1e-6 if zero_centroid else 0.2,
            "affine_drift_added": zero_centroid, "parallax_factor_set_zero": zero_pf,
            "centroid_set_zero": zero_centroid}, "result": P.fit_photocentre(synthetic)}
    return cases


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_report() -> dict:
    source, joint = load_geometry()
    records = [{"source_id": str(int(row.source_id)), "transit_id": str(int(row.transit_id)),
                "phot_time_gaia_d": float(row.t), "astro_time_gaia_d": float(row.t_astro),
                "t_yr": float(row.t_yr), "theta_rad": float(row.theta),
                "pf_al": float(row.pf_al), "sx_al_mas": float(row.sx_al),
                "native_x_al_mas": float(row.x_al), "native_phi": float(row.phi)}
               for row in joint.itertuples()]
    return _json_ready({
        "schema_version": 1, "result_kind": "CONDITIONAL_SYNTHETIC_SPATIAL_CONTROL",
        "ai_authorship": "OpenAI Codex; independent review recorded in handoff",
        "source": source, "native_geometry_rows": records, "affine_drift_control": drift_case(joint),
        "independent_flux_controls": independent_flux_cases(joint),
        "limitations": [
            "Injected flux/centroid are synthetic; native observing geometry/errors are preserved.",
            "Analytic D null directions show conditional model nonidentifiability, not a sky blend.",
            "Raw native flux/centroid remain separate; no native fit or historical output is rewritten.",
            "No empirical contamination rate, astronomical detection or A4/release-gate claim.",
            "Undefined numeric diagnostics are serialized null, never zero.",
        ],
    })


def main() -> None:
    canonical = json.dumps(build_report(), sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n"
    print("SPATIAL_CONTROL_JSON " + canonical.rstrip("\n"))
    print("SPATIAL_CONTROL_REPORT_SHA256 " + hashlib.sha256(canonical.encode()).hexdigest())


if __name__ == "__main__":
    main()

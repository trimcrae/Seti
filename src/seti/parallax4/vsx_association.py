"""VSX/Gaia association audit with a caller-reviewed evidence trust boundary.

Live cone proximity alone is never a target identity. The native VSX columns
RAJ2000/DEJ2000 label equinox coordinates, not a demonstrated position epoch;
the photometric Epoch field is not used as an astrometric epoch.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MAX_CANDIDATES = 100
CATALOGUE = "B/vsx/vsx"
SUPPORTED = "CROSS_ID_RECEIPT_POSITION_CONSISTENT"


def _identifier(value):
    # Opaque exact numeric catalogue keys; integral floats may already be lossy.
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        value = str(value)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value) \
            or not value.strip("0"):
        return None
    return value


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def _json_scalar(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _direction(ra, dec):
    ra, dec = np.deg2rad([ra, dec])
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def propagated_direction(target: dict, epoch: float) -> np.ndarray:
    """Normalized constant tangent-motion approximation, with pmra = mu_alpha*.

    This spherical model handles RA wrap and poles. It does not establish
    perspective acceleration, covariance, long-baseline accuracy or identity.
    """
    values = {k: _number(target.get(k)) for k in ("ra", "dec", "ref_epoch", "pmra", "pmdec")}
    epoch = _number(epoch)
    if epoch is None or any(v is None for v in values.values()):
        raise ValueError("finite Gaia position, reference epoch and proper motion required")
    ra, dec = values["ra"], values["dec"]
    if not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError("invalid Gaia ICRS direction")
    alpha, delta = np.deg2rad([ra, dec])
    east = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    north = np.array([-np.sin(delta) * np.cos(alpha),
                      -np.sin(delta) * np.sin(alpha), np.cos(delta)])
    mas_to_rad = np.deg2rad(1 / 3600000)
    velocity = mas_to_rad * (values["pmra"] * east + values["pmdec"] * north)
    direction = _direction(ra, dec) + (epoch - values["ref_epoch"]) * velocity
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("proper-motion projection is not finite")
    return direction / norm


def receipt_context(target: dict | None, receipt: dict | None):
    """Validate syntax/domain bindings; this does NOT authenticate the receipt."""
    if not isinstance(receipt, dict):
        return None, "CALLER_REVIEWED_RECEIPT_MISSING"
    if receipt.get("schema") != "vsx-gaia-association-v1" \
            or receipt.get("review_status") != "CALLER_REVIEWED" \
            or receipt.get("catalogue_table") != CATALOGUE \
            or receipt.get("gaia_release") != "DR3" \
            or receipt.get("position_frame") != "ICRS":
        return None, "RECEIPT_DOMAIN_OR_REVIEW_UNSUPPORTED"
    evidence_url, evidence_hash = receipt.get("evidence_url"), receipt.get("evidence_sha256")
    if not isinstance(evidence_url, str) or not re.fullmatch(r"https://[^\s]+", evidence_url) \
            or not isinstance(evidence_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
        return None, "RECEIPT_PROVENANCE_INVALID"
    if not isinstance(target, dict) or _identifier(target.get("source_id")) is None:
        return None, "GAIA_ASTROMETRY_OR_IDENTIFIER_MISSING"
    if _identifier(receipt.get("gaia_source_id")) != _identifier(target["source_id"]) \
            or _identifier(receipt.get("vsx_oid")) is None:
        return None, "RECEIPT_IDENTIFIER_MISMATCH"
    epoch = _number(receipt.get("position_epoch_jyear"))
    tolerance = _number(receipt.get("max_separation_arcsec"))
    if epoch is None:
        return None, "POSITION_EPOCH_UNKNOWN"
    if tolerance is None or tolerance <= 0:
        return None, "RECEIPT_POSITION_TOLERANCE_INVALID"
    try:
        direction = propagated_direction(target, epoch)
    except ValueError:
        return None, "GAIA_ASTROMETRY_INCOMPLETE"
    return {"direction": direction, "epoch": epoch, "tolerance": tolerance}, None


def audit_candidates(rows: pd.DataFrame, *, target: dict | None = None,
                     receipt: dict | None = None) -> dict:
    """Retain bounded native rows; assign a type only with a coherent receipt."""
    candidates = [
        {str(k): _json_scalar(v) for k, v in row.items()}
        for row in rows.head(MAX_CANDIDATES + 1).to_dict("records")
    ]
    result = {
        "vsx_status": "ROWS_RETURNED" if len(rows) else "NO_ROWS_AT_QUERY_POSITION",
        "vsx_n_rows_returned": int(len(rows)), "vsx_n_rows_preserved": len(candidates),
        "vsx_query_truncated": bool(len(rows) > MAX_CANDIDATES),
        "vsx_candidates": candidates,
        "vsx_target_astrometry": {k: _json_scalar(v) for k, v in (target or {}).items()
                                 if k in ("source_id", "ra", "dec", "ref_epoch", "pmra", "pmdec")},
        "vsx_association_receipt": receipt,
        "vsx_coordinate_columns": "RAJ2000/DEJ2000: reported J2000 equinox",
        "vsx_position_epoch_status": "UNKNOWN_WITHOUT_CALLER_REVIEWED_EVIDENCE",
        "vsx_association_method": "CALLER_REVIEWED_CROSS_ID_AND_CONSTANT_TANGENT_POSITION",
        "vsx_receipt_authenticated_by_program": False,
    }
    context, reason = receipt_context(target, receipt)
    if result["vsx_query_truncated"]:
        reason = "CONE_RESPONSE_TRUNCATED"
    if reason:
        result["vsx_association_status"] = reason
        return result
    result.update(vsx_position_epoch_jyear=context["epoch"],
                  vsx_position_epoch_status="CALLER_REVIEWED_RECEIPT",
                  vsx_max_separation_arcsec=context["tolerance"])
    if not candidates:
        result["vsx_association_status"] = "NO_ROWS_AT_QUERY_POSITION"
        return result
    ids, compatible, malformed = [], [], False
    for candidate in candidates:
        cols = {k.lower(): k for k in candidate}
        oid = _identifier(candidate.get(cols.get("oid", "")))
        ra = _number(candidate.get(cols.get("raj2000", "")))
        dec = _number(candidate.get(cols.get("dej2000", "")))
        if oid is None or ra is None or dec is None or not 0 <= ra < 360 or not -90 <= dec <= 90:
            malformed = True
            candidate["association_row_status"] = "IDENTIFIER_OR_POSITION_INVALID"
            continue
        ids.append(oid)
        direction = _direction(ra, dec)
        angle = np.arctan2(np.linalg.norm(np.cross(context["direction"], direction)),
                           np.dot(context["direction"], direction))
        separation = float(np.rad2deg(angle) * 3600)
        candidate["association_separation_arcsec"] = separation
        candidate["association_row_status"] = (
            "POSITION_COMPATIBLE" if separation <= context["tolerance"] else "POSITION_OFFSET")
        if separation <= context["tolerance"]:
            compatible.append(candidate)
    result["vsx_n_position_compatible"] = len(compatible)
    if malformed:
        result["vsx_association_status"] = "CANDIDATE_ID_OR_POSITION_INVALID"
    elif len(ids) != len(set(ids)):
        result["vsx_association_status"] = "DUPLICATE_NATIVE_OBJECT_IDS"
    elif len(compatible) != 1:
        result["vsx_association_status"] = (
            "AMBIGUOUS_POSITION_CANDIDATES" if compatible else "POSITION_OFFSET")
    else:
        selected = compatible[0]
        cols = {k.lower(): k for k in selected}
        if _identifier(selected[cols["oid"]]) != _identifier(receipt["vsx_oid"]):
            result["vsx_association_status"] = "RECEIPT_OBJECT_NOT_POSITION_MATCH"
        else:
            result.update(
                vsx_association_status=SUPPORTED, vsx_oid=selected[cols["oid"]],
                vsx_name=selected.get(cols.get("name", ""), ""),
                vsx_type=selected.get(cols.get("type", ""), ""),
                vsx_period=selected.get(cols.get("period", "")),
            )
    return result


def classification_type(row: dict) -> str | None:
    """Recompute bindings and geometry; serialized status/type alone is insufficient."""
    if row.get("vsx_association_status") != SUPPORTED:
        return None
    target = row.get("vsx_target_astrometry")
    if not isinstance(target, dict):
        return None
    if "source_id" in row and _identifier(row["source_id"]) != _identifier(target.get("source_id")):
        return None
    if row.get("vsx_query_truncated") is not False:
        return None
    candidates = row.get("vsx_candidates")
    if not isinstance(candidates, list) or not all(isinstance(x, dict) for x in candidates):
        return None
    audit = audit_candidates(pd.DataFrame(candidates), target=target,
                             receipt=row.get("vsx_association_receipt"))
    if audit.get("vsx_association_status") != SUPPORTED:
        return None
    # Recomputed native value wins, never a forged legacy/serialized type flag.
    return str(audit.get("vsx_type") or "")

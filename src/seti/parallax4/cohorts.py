"""Validate one PARALLAX4 sweep cohort before any reduction output is written."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PREFIXES = ("events", "eventsAB", "injections", "pixel_epochs")
_VERDICTS = {"SHARD_COMPLETE", "SHARD_PARTIAL", "SHARD_PARTIAL_TIME_BUDGET"}


class CohortError(ValueError):
    """Inputs cannot establish a coherent sweep cohort."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_manifest(out: Path, tag: str) -> dict:
    return {prefix: sha256(path) if path.exists() else None
            for prefix in PREFIXES
            for path in [out / f"{prefix}_{tag}.csv"]}


def _integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise CohortError(f"{label}: expected a nonnegative integer")
    if value < 0 or int(value) != value:
        raise CohortError(f"{label}: expected a nonnegative integer")
    return int(value)


def load_cohort(out: Path, *, n_shards_expected: int | None = None,
                run_id_expected: str | None = None) -> dict:
    """Current records have provenance and compact uniform-grid time_edges.
    Legacy CSVs have no run column: validate their filenames, record counts and
    CDN-file lineage. New manifests additionally bind their exact bytes.
    """
    if n_shards_expected is not None and _integer(n_shards_expected, "shards") < 1:
        raise CohortError("shards must be positive")
    records, refused, ignored = [], [], []
    for path in sorted(out.glob("sweep_s*of*.json")):
        match = re.fullmatch(r"sweep_s(\d+)of(\d+)\.json", path.name)
        if match is None:
            raise CohortError(f"noncanonical shard filename: {path.name}")
        shard, width = map(int, match.groups())
        if n_shards_expected is not None and width != n_shards_expected:
            ignored.append(path.name)
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise CohortError(f"invalid shard record: {path.name}") from exc
        if not isinstance(record, dict) or record.get("stage") != "sweep":
            raise CohortError(f"invalid sweep stage: {path.name}")
        if (_integer(record.get("shard"), path.name) != shard
                or _integer(record.get("n_shards"), path.name) != width
                or width < 1 or shard >= width or path.name != f"sweep_s{shard}of{width}.json"):
            raise CohortError(f"shard filename/metadata mismatch: {path.name}")
        for key in ("run_id", "git_sha"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise CohortError(f"{path.name}: missing {key}")
        if run_id_expected is not None and record["run_id"] != str(run_id_expected):
            raise CohortError(f"{path.name}: wrong run_id {record['run_id']!r}; expected {run_id_expected!r}")
        record = {**record, "_tag": f"s{shard}of{width}"}
        if record.get("verdict") == "REFUSED_CONTROLS_NOT_PASSED":
            refused.append(record)
        elif record.get("verdict") in _VERDICTS:
            records.append(record)
        else:
            raise CohortError(f"{path.name}: unsupported sweep verdict {record.get('verdict')!r}")
    all_records = records + refused
    for key in ("n_shards", "run_id", "git_sha"):
        if len({record[key] for record in all_records}) > 1:
            raise CohortError(f"mixed {key} in sweep cohort")
    grid = None
    for record in records:
        tag = record["_tag"]
        te = record.get("time_edges")
        if not isinstance(te, list) or len(te) != 3:
            raise CohortError(f"{tag}: missing uniform histogram grid")
        nb = _integer(te[2], f"{tag} time_edges bins")
        if (nb < 1 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                          or not np.isfinite(x) for x in te[:2]) or te[0] >= te[1]):
            raise CohortError(f"{tag}: invalid histogram grid")
        if grid is not None and te != grid:
            raise CohortError(f"{tag}: mismatched time_edges")
        grid = te
        for key in ("hist_transits", "hist_grey_episodes", "hist_all_episodes"):
            try:
                hist = np.asarray(record.get(key), dtype=float)
            except (TypeError, ValueError) as exc:
                raise CohortError(f"{tag}: invalid {key}") from exc
            if hist.shape != (nb,) or not np.isfinite(hist).all() or (hist < 0).any():
                raise CohortError(f"{tag}: invalid {key}; expected {nb} finite nonnegative bins")
        counts = record.get("counts")
        if not isinstance(counts, dict) or not {"n_events", "n_AB"} <= counts.keys():
            raise CohortError(f"{tag}: missing event counts")
        for key, value in counts.items():
            _integer(value, f"{tag} counts.{key}")
        if counts["n_AB"] > counts["n_events"]:
            raise CohortError(f"{tag}: n_AB exceeds n_events")
        done = record.get("done")
        if (not isinstance(done, list) or any(not isinstance(x, str) or not x for x in done)
                or len(set(done)) != len(done)
                or _integer(record.get("n_files_done"), tag) != len(done)
                or _integer(record.get("n_files_planned"), tag) < len(done)):
            raise CohortError(f"{tag}: invalid processed-file bookkeeping")
        manifest = record.get("output_sha256")
        if "artifact_schema" in record or manifest is not None:
            if record.get("artifact_schema") != 1 or not isinstance(manifest, dict) or set(manifest) != set(PREFIXES):
                raise CohortError(f"{tag}: invalid output manifest")
            for prefix in PREFIXES:
                path = out / f"{prefix}_{tag}.csv"
                expected = manifest[prefix]
                if expected is None:
                    if path.exists():
                        raise CohortError(f"{path.name}: unexpected output (manifest says absent)")
                elif (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                      or not path.exists() or sha256(path) != expected):
                    raise CohortError(f"{path.name}: output checksum mismatch or missing file")
    processed = [name for record in records for name in record["done"]]
    if len(set(processed)) != len(processed):
        raise CohortError("processed CDN file appears in multiple shards")
    tags = {record["_tag"] for record in records}
    widths = {record["n_shards"] for record in all_records}
    for prefix in PREFIXES:
        for path in sorted(out.glob(f"{prefix}_s*of*.csv")):
            match = re.fullmatch(rf"{prefix}_s(\d+)of(\d+)\.csv", path.name)
            if match is None:
                raise CohortError(f"noncanonical output filename: {path.name}")
            shard, width = map(int, match.groups())
            if n_shards_expected is not None and width != n_shards_expected:
                ignored.append(path.name)
                continue
            if widths and width not in widths:
                raise CohortError(f"{path.name}: mixed shard width")
            if f"s{shard}of{width}" not in tags or path.name != f"{prefix}_s{shard}of{width}.csv":
                raise CohortError(f"{path.name}: output has no accepted sweep record")
    first = all_records[0] if all_records else {}
    return {"records": records, "refused": refused, "ignored": ignored,
            "run_id": first.get("run_id"), "git_sha": first.get("git_sha"),
            "n_shards": first.get("n_shards"), "time_edges": grid,
            "legacy_shards": [r["_tag"] for r in records if "artifact_schema" not in r],
            "legacy_csv_provenance": ("unbound bytes; record counts and CDN-file lineage only"
                                      if any("artifact_schema" not in r for r in records)
                                      else "all selected output bytes bound by manifests"),
            "legacy_pixel_completeness": ("unresolved beyond transit/histogram lower bounds"
                                          if any("n_pixel_transits" not in r["counts"] for r in records)
                                          else "checked against exact n_pixel_transits"),
            "unresolved_identity_fields": [key for key in ("run_id", "git_sha")
                                            if first.get(key) in ("local", "unknown")]}


def read_tables(out: Path, cohort: dict, prefix: str, *, required: list[str],
                count_key: str | None = None, usecols: list[str] | None = None) -> list[pd.DataFrame]:
    """Read only accepted tags, reject corrupt or inconsistent companion CSVs."""
    frames = []
    for record in cohort["records"]:
        path = out / f"{prefix}_{record['_tag']}.csv"
        expected = record["counts"].get(count_key, 0) if count_key else None
        if not path.exists():
            if expected:
                raise CohortError(f"{path.name}: missing {expected} recorded rows")
            continue
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise CohortError(f"{path.name}: unreadable CSV") from exc
        if not set(required) <= set(frame.columns):
            raise CohortError(f"{path.name}: missing required columns")
        if expected is not None and len(frame) != expected:
            raise CohortError(f"{path.name}: row count {len(frame)} != recorded {expected}")
        if prefix != "pixel_epochs" and len(frame):
            if "file" not in frame or not frame["file"].isin(record["done"]).all():
                raise CohortError(f"{path.name}: rows outside recorded processed files")
        if prefix == "pixel_epochs":
            for key in required:
                values = pd.to_numeric(frame[key], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
                    raise CohortError(f"{path.name}: invalid {key}")
            total = int(frame["n"].sum())
            exact = record["counts"].get("n_pixel_transits")
            lower = max(record["counts"].get("n_transits_ok_g", 0),
                        float(np.asarray(record["hist_transits"]).sum()))
            if (exact is not None and total != exact) or total < lower:
                raise CohortError(f"{path.name}: pixel transit total disagrees with record")
        frames.append(frame)
    return frames

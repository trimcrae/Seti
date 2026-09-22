"""DASCH AFLAGS / BFLAGS --- the per-detection quality and blending bits.

The bit assignments are not reproduced from memory.  They are parsed from the
``daschlab`` source (``class AFlags(IntFlag)`` / ``class BFlags(IntFlag)``) that
the probe stage fetches from GitHub on the runner, with the parsed result
written into ``results/century/probe/flag_bits.json`` and, once verified, into
``config/century.yaml`` as the offline fallback.  A run that has *neither* a
parsed nor a configured definition applies no flag cuts and says so:
``flags_applied = False`` is a first-class degradation, never a silent default.

Only two facts about the flags are load-bearing for the channel:

* which bits mean **blend** (a neighbour inside the photographic PSF, whose
  contribution changes with plate scale and emulsion and therefore with plate
  *series*, i.e. with calendar time --- the time-clustered blending kill);
* which bits mean **do not use this point at all** (defects, bad plates,
  uncertain dates, saturation).

Everything else is carried through untouched.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

import numpy as np

# Names (case-insensitive substrings) that identify blend bits and reject bits,
# whatever their numeric value turns out to be.  Matched against the enum
# member names parsed from the source.
BLEND_NAME_PATTERNS = ("BLEND", "NEIGHBOR", "NEIGHBOUR", "MULTIPLE_MATCH")
REJECT_NAME_PATTERNS = (
    "DEFECT", "BAD_PLATE", "UNCERTAIN_DATE", "SATURAT", "CORRUPT", "PICKERING",
    "MULT_EXP_UNMATCHED", "LARGE_ISO_RMS", "LARGE_LOCAL_SMOOTH", "RADIAL_BIN_9",
    "BIN_DRAD_UNKNOWN", "LARGE_DRAD", "TOO_BRIGHT", "LOW_ALTITUDE",
    "LARGE_SMOOTHING", "REJECTED", "TRUNCATED", "OVERFLOW",
)


@dataclass
class FlagDefs:
    """Parsed bit definitions for one flag word."""

    name: str
    bits: dict[str, int] = field(default_factory=dict)
    source: str = "none"

    @property
    def available(self) -> bool:
        return bool(self.bits)

    def mask_for(self, patterns) -> int:
        m = 0
        for nm, v in self.bits.items():
            up = nm.upper()
            if any(p in up for p in patterns):
                m |= int(v)
        return m

    @property
    def blend_mask(self) -> int:
        return self.mask_for(BLEND_NAME_PATTERNS)

    @property
    def reject_mask(self) -> int:
        return self.mask_for(REJECT_NAME_PATTERNS)

    def names(self, value: int) -> list[str]:
        return [nm for nm, v in self.bits.items() if int(value) & int(v)]

    def as_dict(self) -> dict:
        return {"name": self.name, "source": self.source, "bits": dict(self.bits),
                "blend_mask": self.blend_mask, "reject_mask": self.reject_mask}


def _eval_const(node) -> int | None:
    """Evaluate ``1 << 6``, ``0x40``, ``64``, ``A | B``-style constant expressions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return int(node.value)
    if isinstance(node, ast.BinOp):
        a, b = _eval_const(node.left), _eval_const(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.LShift):
            return a << b
        if isinstance(node.op, ast.BitOr):
            return a | b
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Pow):
            return a ** b
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "auto":
        return None
    return None


def parse_flag_classes(source: str) -> dict[str, dict[str, int]]:
    """Find every ``class X(IntFlag)``/``(Flag)``/``(IntEnum)`` and its int members."""
    out: dict[str, dict[str, int]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
        if not any(b in ("IntFlag", "Flag", "IntEnum", "Enum") for b in bases):
            continue
        members: dict[str, int] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                v = _eval_const(stmt.value)
                if v is not None:
                    members[stmt.targets[0].id] = int(v)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) \
                    and stmt.value is not None:
                v = _eval_const(stmt.value)
                if v is not None:
                    members[stmt.target.id] = int(v)
        if members:
            out[node.name] = members
    return out


def flagdefs_from_source(source: str) -> tuple[FlagDefs, FlagDefs]:
    """``(aflags, bflags)`` from the daschlab ``lightcurves.py`` text."""
    classes = parse_flag_classes(source)
    a = FlagDefs("aflags")
    b = FlagDefs("bflags")
    for cname, members in classes.items():
        if re.match(r"(?i)^a_?flags?$", cname):
            a.bits, a.source = dict(members), "daschlab_source"
        elif re.match(r"(?i)^b_?flags?$", cname):
            b.bits, b.source = dict(members), "daschlab_source"
    return a, b


def flagdefs_from_config(conf: dict | None) -> tuple[FlagDefs, FlagDefs]:
    """Fallback definitions from ``config/century.yaml`` (``flag_bits`` section)."""
    a = FlagDefs("aflags")
    b = FlagDefs("bflags")
    fb = (conf or {}).get("flag_bits") or {}
    if isinstance(fb.get("aflags"), dict) and fb["aflags"]:
        a.bits = {str(k): int(v) for k, v in fb["aflags"].items()}
        a.source = "config"
    if isinstance(fb.get("bflags"), dict) and fb["bflags"]:
        b.bits = {str(k): int(v) for k, v in fb["bflags"].items()}
        b.source = "config"
    return a, b


def resolve_flagdefs(conf: dict | None = None, source_text: str | None = None
                     ) -> tuple[FlagDefs, FlagDefs]:
    """Parsed source first, config second, nothing third (and say so)."""
    if source_text:
        a, b = flagdefs_from_source(source_text)
        if a.available or b.available:
            ca, cb = flagdefs_from_config(conf)
            return (a if a.available else ca), (b if b.available else cb)
    return flagdefs_from_config(conf)


def apply_masks(aflags: np.ndarray, bflags: np.ndarray, a: FlagDefs, b: FlagDefs
                ) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(is_blend, is_reject, applied)`` for arrays of flag words."""
    af = np.nan_to_num(np.asarray(aflags, dtype=float), nan=0.0).astype(np.int64)
    bf = np.nan_to_num(np.asarray(bflags, dtype=float), nan=0.0).astype(np.int64)
    applied = bool(a.available or b.available)
    blend = ((af & a.blend_mask) != 0) | ((bf & b.blend_mask) != 0)
    reject = ((af & a.reject_mask) != 0) | ((bf & b.reject_mask) != 0)
    return blend, reject, applied


__all__ = ["BLEND_NAME_PATTERNS", "FlagDefs", "REJECT_NAME_PATTERNS", "apply_masks",
           "flagdefs_from_config", "flagdefs_from_source", "parse_flag_classes",
           "resolve_flagdefs"]

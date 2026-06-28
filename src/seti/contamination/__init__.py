"""Contamination-rejection funnel.

Each module exposes a ``*_pass(df, thresholds) -> pd.Series[bool]`` function.
:func:`seti.contamination.run_funnel` applies them in order, recording, for every
source, the first cut it failed (``reject_reason``) so the funnel is fully
auditable and the per-cut survivor counts can be tabulated for the paper.
"""

from __future__ import annotations

import pandas as pd

from . import astrometry, comovement, crowding, extragalactic, wise_quality

# Ordered funnel: (stage name, predicate).  Order matters only for which reason
# is attributed first; the surviving set is order-independent.
FUNNEL = [
    ("astrometry", astrometry.astrometry_pass),
    ("wise_quality", wise_quality.wise_quality_pass),
    ("crowding", crowding.crowding_pass),
    ("comovement", comovement.comovement_pass),
    ("extragalactic", extragalactic.extragalactic_pass),
]


def run_funnel(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Apply every contamination cut; annotate ``reject_reason`` and ``clean``.

    Returns a copy of ``df`` with a string ``reject_reason`` ("" if it survives
    all cuts) and a boolean ``clean`` column.  Also returns per-stage survivor
    counts via the ``funnel_counts`` attribute on the DataFrame's ``attrs``.
    """
    out = df.copy().reset_index(drop=True)
    out.attrs.update(df.attrs)  # .copy() can drop attrs; co-movement needs epochs
    reason = pd.Series([""] * len(out), index=out.index, dtype=object)
    alive = pd.Series(True, index=out.index)

    counts = {"input": int(len(out))}
    for name, predicate in FUNNEL:
        passed = predicate(out, thresholds).reindex(out.index).fillna(False)
        newly_failed = alive & ~passed
        reason.loc[newly_failed] = name
        alive &= passed
        counts[name] = int(alive.sum())

    out["reject_reason"] = reason
    out["clean"] = alive
    out.attrs["funnel_counts"] = counts
    return out


__all__ = ["run_funnel", "FUNNEL"]

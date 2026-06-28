"""Multi-modal technosignature anomaly indicators.

The premise: infrared excess alone is a weak, single-axis anomaly that warm dust
and unresolved companions readily mimic.  A compelling technosignature candidate
is anomalous along *several independent axes at once* -- in particular, it should
satisfy the energy-balance prediction (flux removed at short wavelengths
reappearing as infrared waste heat), and may additionally show optical or
mid-infrared variability, an astrometric companion signature, or anomalous
kinematics.  Each axis is an :class:`Indicator` returning a per-object,
[0, 1]-normalised anomaly score and a boolean flag; :func:`combine_indicators`
counts how many *independent* axes flag each object and ranks the multi-axis
anomalies as the exciting candidates.
"""

from __future__ import annotations

from .base import Indicator, IndicatorResult
from .combine import combine_indicators

__all__ = ["Indicator", "IndicatorResult", "combine_indicators"]

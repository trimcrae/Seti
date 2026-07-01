"""Astrometric-acceleration technosignature search (Gaia DR3 dark companions)."""

from .analyze import analyze_accelerations, rank_candidates
from .run import accel_run

__all__ = ["analyze_accelerations", "rank_candidates", "accel_run"]

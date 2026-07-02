"""Panspermia close-encounter channel.

Anchored on K2-18 (host of the hycean world K2-18 b, JWST biosignature hint), this
channel asks a kinematic question: which stars passed close to K2-18, at low
relative velocity, in the recent past -- and are therefore the systems most likely
to have received K2-18-origin material if life there ever escaped the planet.

The neighbourhood reshuffles over time, so the search is over *closest approach*
in full 6D phase space (Gaia DR3), not present-day proximity; the transfer vector
is unbound ejecta / free-flying bodies, so the filter is encounter geometry
(close + slow), not a continuous bridge.
"""

from .encounters import (
    closest_approach,
    flag_comoving,
    regime_summary,
    transfer_regime,
    transfer_score,
)
from .kinematics import phase_space_6d
from .reachability import crossing_times, destination_quality, rank_targets
from .run import K2_18_SOURCE_ID, panspermia_run, targets_run

__all__ = ["phase_space_6d", "closest_approach", "transfer_score",
           "flag_comoving", "transfer_regime", "regime_summary",
           "crossing_times", "destination_quality", "rank_targets",
           "panspermia_run", "targets_run", "K2_18_SOURCE_ID"]

"""Panspermia DONOR / directed-travel channel anchored on LHS 1140.

This is the mirror of the K2-18 panspermia channel.  There, K2-18 is treated as a
possible *source* of life and we ask which stars could have *received* its material.
Here the premise is the same shape but the anchor is different in kind: LHS 1140 b
is a rocky, temperate, classical-habitable-zone world (Rp ~ 1.73 R_earth), not a
hycean sub-Neptune.  So the donor question becomes:

    *If life arose on LHS 1140 b, which reachable nearby worlds could it have
    seeded, and which stars did LHS 1140 pass near?*

Two complementary answers, both computed from Gaia DR3 6D phase space:

* **Recipients** -- the stars LHS 1140 passed CLOSE and SLOW to in the recent past
  (linear closest approach in 6D).  These are the systems into which unbound ejecta
  / free-flying bodies could passively have been delivered.  Reuses the K2-18
  close-encounter engine (:mod:`seti.panspermia`) with LHS 1140 as the anchor.

* **Destinations** -- where a *technological* disperser evolved on LHS 1140 b would
  actually aim.  A traveller from a rocky HZ world seeks OTHER rocky HZ worlds, so
  this channel uses the **classical** (Earth-analog) destination prior -- the exact
  contrast with K2-18's hycean prior -- ranking reachable known-planet hosts by
  destination quality and closest-approach distance.

The maths is the reused, unit-tested panspermia/reachability machinery; only the
anchor and the habitability prior change.
"""

from .run import LHS1140_SOURCE_ID, lhs1140_origin_run

__all__ = ["lhs1140_origin_run", "LHS1140_SOURCE_ID"]

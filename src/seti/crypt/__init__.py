"""CRYPT — artifacts in lunar permanent shadow (signature S55, docs/crypt.md).

A permanently shadowed region (PSR) floor sits at 25–110 K and its regolith
turns over at metres per gigayear: it is the one place near us where an old
object survives, and a *thermally active* one is visible against nothing.  A
sub-pixel warm component inside a 240 m Diviner pixel raises the short
thermal channels (6, 7: 13–41 µm) far more than the long ones (8, 9:
50–400 µm) — the anisothermality that Diviner rock-abundance retrievals use
at low latitudes and that has never been searched inside PSRs.  A season-
independent anisothermal excess (present in the winter product as well as the
summer one) is an internal heat source, not a lit rim or scattered light.

Modules
-------
``labels``   PDS3 / PDS4 label parsing, IMG reading, polar-stereographic and
             equirectangular georeferencing (pure functions, offline).
``thermal``  band radiance, brightness temperature, the PSR (cold-trap) mask,
             the anisothermality statistic with its empirical noise floor, the
             two-component fit, the seasonal / spatial / stripe rejection rules
             and signal injection (pure functions, offline).
``radar``    Mini-RF circular-polarisation-ratio compact-anomaly screen.
``acquire``  PDS Geosciences directory crawl, product classification, capped
             downloads, ODE REST footprint queries (runner only; scripted
             fetcher for tests).
``vet``      the kill ledger applied to every flagged pixel: human hardware,
             radar reading, optical coverage.
``run``      stage orchestration (probe / acquire / screen / assess / all,
             ``--shard i/n``), writes ``results/crypt/``.
"""

from __future__ import annotations

__all__ = ["acquire", "labels", "radar", "run", "thermal", "vet"]

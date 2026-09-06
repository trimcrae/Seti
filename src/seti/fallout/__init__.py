"""FALLOUT — the fission-product abundance *pattern* in a stellar photosphere.

Signature S14 of ``docs/necrosignatures.md``. The sibling channel ``midden``
searches for the radionuclide **lines** themselves (Tc I, Pm II, U/Th) — the
< 10^5 yr window of Whitmire & Wright 1980. FALLOUT is its complement: the
long-lived **shape** that fission product leaves in the n-capture elements
after every short-lived member has decayed, which is the only part of the
signature that outlives its makers by megayears.

Fission yields are two-humped (A~90-105 and A~133-145) with a deep valley
between and nothing past A~155. Folded against solar abundances that is a
fixed vector: ``[Nd/Ba] >> 0``, ``[Ce/Ba] > 0``, ``[La/Ba] > 0``,
``[Mo/Zr] > 0``, ``[Ru/Zr] > 0``, ``[Eu/Nd] < 0`` — a combination neither the
s-process (which makes Ba and Zr well: ``[Nd/Ba] < 0``, ``[Mo/Zr] < 0``) nor the
r-process (which makes Eu: ``[Eu/Nd] > 0``) nor any mixture of them can
produce. Each star's vector is fitted as solar + {s, r, s+r, fission} and the
statistic is the fission-only likelihood ratio against the best natural
mixture. The decisive vet is leave-one-out: if any single element carries the
preference, it is not a pattern.

Modules
-------
``yields``    the physics tables (U-235 chain yields, solar abundances, s/r/p
              fractions) and the three template patterns derived from them.
``pattern``   the fitter, the classification, the vetoes, the leave-one-out
              test, the shuffled-element null and the sensitivity curve.
``acquire``   GALAH DR4 / APOGEE DR17 via ``seti.tailings.acquire`` with the
              runner-proven route first and the extra columns this channel
              needs (``flag_sp``, age, binary flag) discovered at runtime.
``run``       stages ``probe``, ``acquire``, ``screen``, ``assess``; writes
              ``results/fallout/``.

See ``docs/fallout.md``.
"""

from __future__ import annotations

__all__ = ["acquire", "pattern", "run", "yields"]

"""TAILINGS — the sparse chemical anomaly.

Natural stellar abundance space is **low-dimensional**. Every star is a mixture
of a small number of nucleosynthetic sources, and each source moves an entire
element *family* together: the alpha elements track core-collapse supernovae,
the Fe-peak tracks Type Ia, the s-process tracks AGB pollution, the r-process
tracks neutron-star mergers. That is why chemical-tagging studies keep finding
that ~30 measured abundances collapse onto a handful of independent axes.

Industrial refining does not work like that. A refinery separates **one
element** — it is defined by the fact that it does. So the artificial signature
is a **sparse** anomaly: one or two elements extreme, everything else normal.
The natural signature is a **dense** anomaly: a whole family moves at once.
Sparsity, not amplitude, is the discriminant, and it is the thing no published
abundance-anomaly search has used.

The population is chosen to close the last natural loophole. Single-element
peculiarity *does* occur — in Ap, Am and HgMn stars — but it is made by atomic
diffusion and radiative levitation in radiative envelopes, and it switches off
once the convective envelope becomes massive enough to overwhelm the separation
timescale. Restricting to **G/K/M dwarfs (Teff < 6000 K, log g > 4.0)** puts
the search in the regime where the known production mechanism for a
single-element anomaly does not operate.

Two readings of a detection, same observable:

* **tailings** — the residue of planetary disassembly, dumped into the host
  star because a star is the cheapest waste sink in a system; or
* **a tombstone** — a deliberate marker. A photospheric abundance flag is the
  cheapest galaxy-visible, Myr-durable thing a dying civilization could leave.

Modules
-------
``acquire``   GALAH DR4 / APOGEE DR17 / LAMOST pulls and the wide-binary
              catalogue (runner-only; every path records its provenance).
``manifold``  the natural manifold: regress each [X/Fe] on ([Fe/H], Teff,
              log g, an alpha proxy) and keep the empirical residual scatter
              sigma_X(SNR, Teff).
``sparse``    the sparse-anomaly statistic and its explicit dense contrast.
``twins``     stage 4 — co-natal wide binaries whose refractory difference
              exceeds any plausible engulfed-planet mass budget.
``vet``       the contamination funnel, including the raw-spectrum
              re-measurement statistic that any survivor must pass.
``validate``  the validation target — inject Griffith et al. 2021's fifteen
              Na-enhanced stars (arXiv:2110.06240) into a synthetic GALAH DR3
              and measure what the statistic actually recovers. Offline.
``run``       stage orchestration; writes ``results/tailings/``.

See ``docs/tailings.md`` for the claim, the novelty adjudication and the
contamination ledger.
"""

from __future__ import annotations

__all__ = ["acquire", "manifold", "run", "sparse", "twins", "validate", "vet"]

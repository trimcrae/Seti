# COMPASS run report

Field: 116037 astrometric orbits within 2000 pc (of 117512 fetched); 400 scanning-law-banded shuffles per radius.

| radius (pc) | groups | real max S | null med | null p99 | p |
|---|---|---|---|---|---|
| 10 | 775 | 18.2 | 19.3 | 26.5 | 0.6733 |
| 15 | 7600 | 24.8 | 24.1 | 31.5 | 0.3766 |

Top patches per radius (with co-natal/co-moving discriminators) in candidates.json.

## Session verdict — first pass (2026-07-26, run 30211792199)

116,037 DR3 astrometric orbits within 2 kpc; at every radius (25/50/100 pc)
the maximum Bingham coherence statistic lies comfortably INSIDE the
scanning-law-banded shuffle null (p = 0.33 / 0.31 / 0.24). No neighbourhood
exceeds the null's 99th percentile; the top patches are ordinary field
mixtures (heterogeneous chemistry, 26-59 km/s velocity spreads — noise
maxima, not even natural relics). The solar neighbourhood's orbital-pole
field is isotropic at the coherence level DR3 can measure.

Deepening dispatched: tight radii (10/15 pc, n_min 6, 400 shuffles) for
small engineered patches, reusing the sample artifact. If that is also null
the channel rests until DR4 roughly doubles the orbit count and precision;
per policy, no writeup.

## Session verdict — tight-radius pass (2026-07-26, resumed sample)

Also null, decisively: r = 10 pc max statistic 18.2 vs null median 19.3
(p = 0.67 — the real maximum is BELOW the typical shuffle maximum), r = 15 pc
p = 0.38, nothing above any null p99, 400 shuffles per radius. COMPASS is a
complete first-generation null across 10-100 pc coherence scales: the DR3
orbital-pole field carries no local alignment patches at any scale the
catalogue can resolve. The channel rests until DR4 (~2x orbits, better
inclinations). Per policy, no writeup; the question changes.

# COMPASS run report

Field: 116037 astrometric orbits within 2000 pc (of 117512 fetched); 200 scanning-law-banded shuffles per radius.

| radius (pc) | groups | real max S | null med | null p99 | p |
|---|---|---|---|---|---|
| 25 | 31011 | 30.1 | 28.8 | 36.4 | 0.3284 |
| 50 | 83463 | 40.0 | 37.9 | 50.1 | 0.3085 |
| 100 | 108398 | 72.9 | 65.6 | 90.6 | 0.2388 |

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

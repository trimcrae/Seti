# ISOTHERM run report

Search on the SHAPE of infrared excess in temperature space —
emissivity index beta, temperature-distribution width, silicate-feature
equivalent width, and component multiplicity in geometric progression
(docs/isotherm.md).

**Archive verdict:** NOT_PROBED  
**CASSIS reachable:** False  
**Channel verdict:** no_shape_anomaly_in_corpus

## Funnel

| stage | n |
|---|---|
| n_corpus | 5480 |
| n_analysed | 5480 |
| n_screened_out_stage1 | 0 |
| n_full_shape_analysis | 5480 |
| n_s5_isothermal | 0 |
| n_s6_cascade | 0 |
| n_rejected_natural | 0 |

## Candidates: 0

(none at the current thresholds)

## Cascade sensitivity (injection-recovery)

| T ratio | SNR | n recovered | dBIC(discrete-gradient) | separable |
|---|---|---|---|---|
| 1.5 | 60 | 2 | -1.2 | False |
| 1.5 | 150 | 2 | -0.6 | False |
| 1.5 | 400 | 2 | +13.4 | False |
| 1.5 | 1000 | 3 | -11.8 | True |
| 2 | 60 | 2 | +5.2 | False |
| 2 | 150 | 3 | +3.3 | False |
| 2 | 400 | 3 | +3.4 | False |
| 2 | 1000 | 3 | -105.4 | True |
| 2.5 | 60 | 3 | +1.2 | False |
| 2.5 | 150 | 3 | -90.7 | True |
| 2.5 | 400 | 3 | -594.7 | True |
| 2.5 | 1000 | 3 | -3502.5 | True |
| 3 | 60 | 3 | -350.7 | True |
| 3 | 150 | 3 | -2046.0 | True |
| 3 | 400 | 3 | -15287.8 | True |
| 3 | 1000 | 3 | -95635.9 | True |

## Limitations

- 5-38 micron constrains beta only for components whose Wien peak is in band, i.e. T ~ 130-1000 K; outside that window beta and T are degenerate and beta is reported as unconstrained.
- Three Wien peaks inside 5-38 micron force a cascade temperature ratio <= 2.8; at ratio ~2 the per-component width cannot be bounded below the natural floor, so the narrowness tier is unreachable without far-IR photometry.
- The redshift scan loses PAH 6.2/7.7 above z ~ 1.5-3.9, so high-z interlopers are not excluded by the spectral test alone.
- A single-component beta is biased towards 0 for multi-temperature sources; beta is therefore read off the model the data select.

No-null rule (CLAUDE.md): an empty candidate list is a statement about
THIS corpus at THESE thresholds, never a publishable result. The next
moves are extending the baseline with AKARI FIS / IRAS far-IR
photometry (which re-opens the per-component narrowness tier), and the
IRAS LRS Calgary atlas that Carrigan 2009 used — now anchorable to Gaia
parallaxes, which is exactly what his distance/luminosity degeneracy
lacked.

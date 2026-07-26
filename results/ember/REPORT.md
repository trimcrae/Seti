# EMBER — mid-infrared waste heat that switched off

**Verdict:** `NO_SURVIVOR`

## Epoch pairs

| pair | verdict |
|---|---|
| `I12->W3` | usable |
| `I25->W4` | usable |
| `I12->S9W` | usable |
| `I25->L18W` | usable |
| `S9W->W3` | usable |
| `L18W->W4` | usable |

## Funnel

| stage | n |
|---|---|
| acquired | 412914 |
| with_early_photometry | 0 |
| scored | 412914 |
| shortlist | 0 |
| survivors | 0 |

## Rejections by contamination stage

| rule | killed |
|---|---|

## Shortlist

No source exceeded the empirically calibrated threshold.

## Honest sensitivity statement

NEOWISE carries W1/W2 only, so no epoch after 2010 measures 12-25 micron flux; the decades-long baseline exists exclusively between IRAS (1983), AKARI (2006-07) and WISE (2010). Sensitivity is set by the early epoch: IRAS reaches ~0.4 Jy at 12 micron, so the 27-year pair probes only very large excesses. The published cross-epoch stability floor for IRAS-to-WISE is 4 percent (HD 172555), and no fade smaller than a few times that is believable regardless of its formal significance.

The detection threshold is set by the sample's own *rising* tail, not by a Gaussian assumption: every symmetric systematic populates fades and rises equally, so only the excess of faders over risers above the threshold can contain signal. The one asymmetric systematic — the flux-limited Eddington bias of the early epoch, which fades only — is corrected explicitly using the source-count slope measured from the catalogue itself.

A null here is not a result and is not to be written up as one. The informative quantity it would produce — the first measurement of the rate of mid-infrared excess appearance and disappearance at 12–25 micron — is recorded internally as an honesty check on the funnel.

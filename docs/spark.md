# SPARK — a single spectral element in excess on a stellar point source

S48 (SPHEREx QR2 spectral images) and S49 (Euclid Q1 NISP line features) of
`docs/necrofrontier.md`. One observable, two surveys: **a star whose spectrum
carries flux in one resolution element that nothing in the star, nothing in a
background galaxy and nothing in the instrument accounts for.**

No SPHEREx or Euclid technosignature search exists. Every published optical
laser search (Tellis & Marcy; Breakthrough Listen; this repository's own
SDSS/DESI `spectra` channel) stops near 0.98 µm. Both surveys work in the band
where our own high-power lasers live — Nd:YAG 1.064 µm, Yb-fibre 1.03–1.09,
Er-fibre 1.53–1.57 µm — and those wavelengths are carried here as
**descriptors, never as filters**: the search is blind across the whole band
and the flag is reported for the reader (`docs/roman.md` §2.2).

---

## 1. What the two surveys actually serve

Verified from `results/necrofrontier/probe.json` (IRSA TAP, probe run 3) and
re-verified on every run by `seti spark --stage probe` →
`results/spark/probe.json`.

| | Euclid Q1 (S49) | SPHEREx QR2 (S48) |
|---|---|---|
| Product | `euclid_q1_spe_lines_line_features`, joined in-archive to `euclid_q1_mer_catalogue` | level-2 spectral images through `spherex.obscore` / `spherex.plane` × `spherex.artifact` |
| Scale | 4,313,551 sources with a 1-D red-grism spectrum | 1,367,432 public spectral-image products |
| Band, resolution | 1.21–1.89 µm, R ≈ 450 | 0.75–5 µm, R ≈ 35–130, 6.15″ pixels |
| The feature is | a fitted line: `spe_line_central_wl_gf`, `spe_line_snr_gf`, `spe_line_fwhm_gf`, `spe_line_ew_gf` | a per-channel residual we measure ourselves by forced aperture photometry |
| Repeat handle | `spe_line_n_dith` — the number of dithers the line was measured in | ≥ 2 sky passes and ≥ 2 detector positions |

The line table's `spe_line_name` is the pipeline's identification *under that
redshift rank's solution*, so the channel treats the name as a descriptor and
re-derives every identification from the observed wavelength alone
(`src/seti/spark/lines.py`). A star carrying a feature the pipeline labelled a
galaxy line at some z is exactly the object this search is looking for.

---

## 2. The statistic

**S49 (Euclid).** Pull one declination strip of one Q1 field at a time
(`strip_adql`), splitting a strip that truncates at `maxrec`. Deduplicate the
same feature across redshift ranks (highest S/N wins; the alternative names are
kept). Cross-match to Gaia DR3 propagated to the Euclid epoch, keeping only
`parallax_over_error > 5`. Then the veto ladder (`VETO_NAMES`), every rung a
boolean column on the survivor table:

`low_snr` · `few_dithers` (`spe_line_n_dith` ≥ 2 where the table exposes it) ·
`band_edge` · `broad` (FWHM > 2.5 resolution elements: an unresolved feature
only) · `non_positive_flux` · `stellar_line` (discrete lines only — at R ≈ 450
the FWHM test already separates a molecular band) · `galaxy_pattern` (one
redshift at which ≥ 2 of the object's features are known galaxy lines:
Hα, Hβ, [O III], [O II], [N II], [S II], Lyα, Mg II, C IV, …) · `blend`
(a Gaia neighbour within 6″ brighter than G+2) · `not_point_like` ·
`spurious` · `recurrent_wavelength` (the same observed wavelength on ≥ 3
different stars is the instrument, not the sky).

Descriptors that never veto: `industrial_flag`, `stellar_band_name`,
`single_line_z` (the redshift each strong emitter would imply for a lone
feature), `n_dispersion_neighbours` (sources along the ~135″ RGS trace).
Trials: N(stars with a spectrum) × N(resolution elements in band).

**S48 (SPHEREx).** Gaia seeds in a 15′ box (G 9–15, `parallax_over_error > 5`,
isolated: no neighbour brighter than G+3 within 20″, none brighter than G+6
within 9″). Every level-2 product covering the box → one IBE cutout per image
serving every seed in it. Forced aperture photometry (r = 2 px, annulus 4–7 px,
variance plane, ZODI plane subtracted, any non-zero FLAGS pixel in the aperture
drops the sample). Wavelength per pixel from the product's own `WCS-WAVE`
table. Per star and detector, a clipped polynomial-plus-running-median
continuum, then per-channel combined z. A candidate needs z ≥ 5 **and** ≥ 2
passes above 2.5σ **and** ≥ 2 detector positions above 2.5σ **and** both
neighbouring channels quiet (|z| ≤ 2) **and** ≥ 2 resolution elements from a
detector band edge. Then the same cross-star vetoes: stellar line, the same
channel hot on ≥ 3 stars, the same detector pixel hot on ≥ 2.

### 2.1 The stellar-line veto at R ≈ 40, and what it costs

At SPHEREx's low-resolution detectors one resolution element is ~0.03 µm at
1.3 µm, and the H I Brackett series is denser than that. A ±1-resel veto around
every catalogued stellar feature therefore removes most of the band:
`lines.clean_channel_fraction` reports **6–20 % of channels clean at R = 40**
(D3 and D6: none), rising with resolution. That number travels with the
verdict, because a detection can only be claimed in a channel where one was
possible.

One rule follows from the same arithmetic: a **broad molecular band vetoes only
at its edges**. The H₂O 1.4 µm band is five resolution elements wide; it cannot
produce an excess confined to one channel with both neighbours quiet — the
adjacent-channel test already rejects that morphology — but the sharp gradient
at a band edge can leave a one-channel residual. Vetoing a band's whole
interior would delete the SPHEREx D2 band for no physical reason. The band a
surviving channel sits inside is kept as `stellar_band_name`, a descriptor.

---

## 3. Verdicts

`results/spark/summary.json` carries one of

* `NO_DATA_REACHED` — no star was tested in either survey;
* `NO_SPARK_SURVIVOR` — stars were tested, nothing survived the ladder;
* `SPARK_CANDIDATES_PENDING_VET` — ≥ 1 survivor, and the vet is the next step;

prefixed `DEGRADED (...)` whenever a route failed, a strip or box was lost, or
a veto could not be applied. A failed acquisition is never a statement about
the sky.

---

## 4. Running it

```
seti spark --stage probe                       # what the archives serve
seti spark --stage euclid  --shard 0/8         # (field, dec-strip) units
seti spark --stage spherex --shard 0/4         # seed boxes
seti spark --stage assess                      # merge shards, trials, summary
```

`.github/workflows/spark.yml` runs the same four stages on a runner:
`probe → {euclid × N, spherex × M} → assess`, checkpointed per unit and per
box, shard outputs uploaded with `if: always()`, ledgers committed back to the
branch. Thresholds live in `config/spark.yaml`; nothing is hard-coded in the
detectors.

---

## 5. Results

*(filled by the runs; see `results/spark/summary.json` and
`results/spark/candidates.csv`)*

---

## 6. The vet a survivor must pass

`SPARK_CANDIDATES_PENDING_VET` is not a detection. Before any survivor is
believed, in this order:

1. **Re-extract from the pixels.** For Euclid, pull the 1-D spectrum (and,
   where available, the 2-D cutout) of the object and confirm the feature is
   in the data and not an artefact of the fit: check it in each dither
   separately, and check the trace for a neighbour's zeroth order at the
   predicted offset (`docs/roman.md` §2.2). For SPHEREx, re-run the photometry
   with a different aperture and annulus and on the raw (non-ZODI-subtracted)
   plane.
2. **The same star, a different epoch.** Euclid Q1 dithers are minutes apart;
   SPHEREx passes are months apart. A feature that is present in one pass and
   absent in another is either variable or spurious, and the two are separated
   only by the noise in the non-detection.
3. **The same sky, a different instrument.** 2MASS/WISE photometry (SPLICES
   carries both), Gaia XP where the wavelength falls below 1.05 µm, and any
   archival NIR spectrum of the star.
4. **The star itself.** Gaia `ruwe`, `phot_variable_flag`, the colour-magnitude
   position, and whether the field is a star-forming region (the Q1 field
   LDN 1641 is YSO-rich: a survivor there is suspect by construction).
5. **The industrial flag last.** It is read only after the survivor has passed
   1–4, and it is never a reason to promote a candidate — only a thing to
   report.

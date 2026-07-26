# MIDDEN run report

Survey-scale search for short-lived radionuclides (Tc I resonance
triplet; U II / Th II actinides) in ESO Phase-3 HARPS+FEROS spectra —
the Whitmire & Wright (1980) nuclear-waste-disposal technosignature
(docs/midden.md).

Corpus: 178 spectra of 99 stars (49 stars with >= 2 epochs).

## Per-line flag rates (census z >= 4)

| line | role | measured | flagged | rate |
|---|---|---|---|---|
| U II 3859.57 | radionuclide | 178 | 3 | 0.0169 |
| Th II 4019.13 | radionuclide | 178 | 5 | 0.0281 |
| Fe I 4045.81 | rv_ref | 178 | 8 | 0.0449 |
| Fe I 4063.59 | rv_ref | 178 | 7 | 0.0393 |
| Fe I 4071.74 | rv_ref | 178 | 5 | 0.0281 |
| Fe I 4132.06 | rv_ref | 178 | 9 | 0.0506 |
| Fe I 4143.87 | rv_ref | 177 | 8 | 0.0452 |
| DUMMY 4152.30 | control | 178 | 11 | 0.0618 |
| Fe I 4202.03 | rv_ref | 178 | 9 | 0.0506 |
| DUMMY 4222.10 | control | 178 | 13 | 0.0730 |
| Tc I 4238.19 | radionuclide | 178 | 7 | 0.0393 |
| Fe I 4250.79 | rv_ref | 178 | 5 | 0.0281 |
| Tc I 4262.27 | radionuclide | 178 | 6 | 0.0337 |
| Fe I 4271.76 | rv_ref | 178 | 8 | 0.0449 |
| DUMMY 4288.40 | control | 178 | 3 | 0.0169 |
| Tc I 4297.06 | radionuclide | 178 | 5 | 0.0281 |
| Fe I 4325.76 | rv_ref | 178 | 6 | 0.0337 |
| Fe I 4383.55 | rv_ref | 178 | 13 | 0.0730 |
| Fe I 4404.75 | rv_ref | 178 | 12 | 0.0674 |

## Candidates: 1

- **renson_217522** (epochs 1/1, tc_coherent=True): TcI_4238=4.00, TcI_4262=2.74, TcI_4297=2.78, UII_3860=-2.50, ThII_4019=0.69

## Prior-claim anchors

- HD 101065: candidate=False, control_veto=True, TcI_4238=4.99, TcI_4262=9.52, TcI_4297=-5.42, UII_3860=-2.86, ThII_4019=-6.60

No-null rule (CLAUDE.md): an empty candidate list here is a domain
statement about THIS corpus (HARPS+FEROS Phase-3, these line windows,
these thresholds), never a publishable result. Next moves are the
UVES/ESPRESSO collections, the Pm II line set, and epoch-resolved
decay-curve tests on any near-threshold star.

## Session verdict (full-corpus run, 2026-07-26)

- Corpus: 178 unique-epoch HARPS/FEROS spectra of 99 stars (all that the ESO
  Phase-3 archive holds for the 16,363-target list after per-epoch dedup; the
  earlier 92-spectrum result came from discovery truncated at the TAP output
  cap and is superseded).
- Population level: the DUMMY control wavelengths flag at 11/178 and 13/178 —
  higher than every true Tc line (7, 6, 5 per 178). The Tc lines fire at the
  rate expected of arbitrary wavelengths in a blend-heavy CP-star corpus:
  no population-level Tc excess exists in this sample.
- Positive control: Przybylski's star (HD 101065) reaches z(Tc I 4262) = 9.5
  and is correctly control-vetoed — the rare-earth-forest confuser is caught
  by design, not by luck.
- Two of the three interim flags from the truncated corpus (HD 154708,
  gaia_6540158877300621696) did not survive the deeper census baseline.
- Sole surviving flag: HD 217522 (roAp), single epoch, coherent Tc triplet
  z = 4.0 / 2.7 / 2.8, no control veto, z(U II) = -2.5. Assessment: consistent
  with rare-earth line blending in a roAp atmosphere (the historical false-Tc
  class); the negative actinide z is a blend symptom, not a waste signature.
  NOT claimed as a detection.
- Queued follow-up (next work item): HD 217522 deep-dive — pull every ESO
  spectrum of the star across instruments (UVES/X-shooter/HARPS regardless of
  the band-limited discovery), measure the triplet per epoch (Tc at 4.2 Myr
  half-life must be static; roAp blends modulate), and score against a
  Teff/logg-matched roAp comparison sample.

# MIDDEN run report

Survey-scale search for short-lived radionuclides (Tc I resonance
triplet; U II / Th II actinides) in ESO Phase-3 HARPS+FEROS spectra —
the Whitmire & Wright (1980) nuclear-waste-disposal technosignature
(docs/midden.md).

Corpus: 92 spectra of 54 stars (25 stars with >= 2 epochs).

## Per-line flag rates (census z >= 4)

| line | role | measured | flagged | rate |
|---|---|---|---|---|
| U II 3859.57 | radionuclide | 92 | 0 | 0.0000 |
| Th II 4019.13 | radionuclide | 92 | 1 | 0.0109 |
| Fe I 4045.81 | rv_ref | 92 | 0 | 0.0000 |
| Fe I 4063.59 | rv_ref | 92 | 1 | 0.0109 |
| Fe I 4071.74 | rv_ref | 92 | 0 | 0.0000 |
| Fe I 4132.06 | rv_ref | 92 | 0 | 0.0000 |
| Fe I 4143.87 | rv_ref | 91 | 0 | 0.0000 |
| DUMMY 4152.30 | control | 92 | 5 | 0.0543 |
| Fe I 4202.03 | rv_ref | 92 | 1 | 0.0109 |
| DUMMY 4222.10 | control | 92 | 1 | 0.0109 |
| Tc I 4238.19 | radionuclide | 92 | 3 | 0.0326 |
| Fe I 4250.79 | rv_ref | 92 | 0 | 0.0000 |
| Tc I 4262.27 | radionuclide | 92 | 10 | 0.1087 |
| Fe I 4271.76 | rv_ref | 92 | 0 | 0.0000 |
| DUMMY 4288.40 | control | 92 | 1 | 0.0109 |
| Tc I 4297.06 | radionuclide | 92 | 4 | 0.0435 |
| Fe I 4325.76 | rv_ref | 92 | 0 | 0.0000 |
| Fe I 4383.55 | rv_ref | 92 | 0 | 0.0000 |
| Fe I 4404.75 | rv_ref | 92 | 0 | 0.0000 |

## Candidates: 3

- **renson_154708** (epochs 1/1, tc_coherent=False): TcI_4238=11.27, TcI_4262=4.88, TcI_4297=1.29, UII_3860=0.11, ThII_4019=-1.15
- **renson_217522** (epochs 1/1, tc_coherent=True): TcI_4238=3.47, TcI_4262=4.19, TcI_4297=3.52, UII_3860=-2.50, ThII_4019=0.54
- **gaia_6540158877300621696** (epochs 1/1, tc_coherent=True): TcI_4238=3.38, TcI_4262=4.58, TcI_4297=3.49, UII_3860=-2.08, ThII_4019=0.37

## Prior-claim anchors

- HD 101065: candidate=False, control_veto=True, TcI_4238=4.37, TcI_4262=14.79, TcI_4297=-5.95, UII_3860=-2.88, ThII_4019=-5.92

No-null rule (CLAUDE.md): an empty candidate list here is a domain
statement about THIS corpus (HARPS+FEROS Phase-3, these line windows,
these thresholds), never a publishable result. Next moves are the
UVES/ESPRESSO collections, the Pm II line set, and epoch-resolved
decay-curve tests on any near-threshold star.

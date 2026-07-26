# MIDDEN deep-dive: HD 217522

Panel: 12 roAp/cool-Ap stars, 302 epochs; target epochs: 20.

Coherent-triplet epochs: 0/20; panel percentile (median Tc quad z): 0.83

## Stability (chi^2 against constant depth)

- Tc I 4238.19: 20 epochs, chi2/dof 0.10, p(constant) 1, depth span 0.0181
- Tc I 4262.27: 20 epochs, chi2/dof 0.03, p(constant) 1, depth span 0.0125
- Tc I 4297.06: 20 epochs, chi2/dof 0.11, p(constant) 1, depth span 0.0188

## Panel standing (median Tc quadrature z per star)

- HD 101065: 4.92
- HD 176232: 3.33
- HD 217522: 2.18 <-- target
- HD 19918: 1.83
- HD 137949: 0.89
- HD 201601: 0.83
- HD 122970: 0.55
- HD 42659: 0.48
- HD 83368: 0.42
- HD 24712: 0.42
- HD 60435: 0.31
- HD 128898: 0.00

## Session verdict (2026-07-26)

The survey flag does not survive. Against 302 same-class epochs (12 roAp/cool-Ap
stars, same instruments, per-instrument census):

- **0 of 20** HD 217522 epochs show a coherent Tc triplet (the survey's single
  HARPS epoch was ranked against a thin 178-spectrum mixed census; ranked
  against its own class it is unremarkable).
- Panel standing: 83rd percentile (rule requires >90th), *below* Przybylski's
  star (4.92) and 10 Aql (3.33) — both canonical rare-earth-forest objects
  with no Tc claim surviving in the modern literature.
- Depth stability is consistent with constant (chi2/dof 0.03-0.11), but with
  zero coherent epochs and sub-threshold panel standing, constancy is simply
  what stable rare-earth blends also look like at these SNRs.

Attribution: rare-earth line-forest blending at the Tc I wavelengths, the
historical false-Tc mechanism in Ap stars. MIDDEN is now closed at ESO-archive
depth as a fully adjudicated null across all three tiers (population census,
control system, targeted deep-dive). Per project policy this is a reason to
change the question, not a writeup.

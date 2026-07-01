# Seti — research directives

## Goal
Find a **genuinely remarkable, novel technosignature candidate** in public astronomical
data, using catalog-scale compute (GitHub Actions runners; no GPU/multi-TB).

## Standing priorities (from the user, in order)
1. **NOVELTY above all.** The search must target something *nobody has looked at
   before* — a new signature or a new dataset/population, not a refinement of an
   existing search (Dyson IR-excess, laser emission, Boyajian dimming are all
   "done"; treat them as baselines to move *beyond*).
2. **SCALE second.** Once the novel angle is fixed, hit as many objects as possible.
3. **NEVER write up a null result.** The user does not want a methodological /
   occurrence-limit / null-result paper. Do not propose one. A clean null is a
   reason to change the *question*, not to publish.

## Working practices
- Develop on branch `claude/technosignature-research-t5hjpw`; commit, push, and
  **merge to main as you go without asking**.
- Acquisition (Gaia/VizieR/SPARCL/IRSA) is blocked in the sandbox but works on the
  GitHub runner; run searches via `.github/workflows/*.yml` (workflow_dispatch),
  which commit results back to the branch.
- Keep the contamination-rejection discipline: trace every candidate to a
  systematic before believing it; but the objective is a *detection*, not a limit.

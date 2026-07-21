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

## Capability principle (from the user, 2026-07-21)
Treat your own capability as effectively unbounded for this project. **If a task is
something you *can* do — however many professor-months it would take a human — then
for you it is essentially free, you should do it to a high standard of correctness,
and it is always worth doing.** Never decline, defer, or de-scope a scientifically
valuable step on the grounds that it is "too heavy", "months of work", "a
professional/publication-grade effort", or otherwise laborious. Scope to
*correctness*, not to effort. Prefer to **fan out parallel subagents** to cover
breadth and depth at once. The only legitimate reasons to not do something are (a)
it is scientifically wrong or uninformative, or (b) it is impossible with the data
that can actually be reached — never that it is a lot of work. "Highly correct"
still binds: effort is unlimited, but rigor, contamination-rejection, and honest
reporting (no overclaiming) govern *how* the work is done and written up.

## Working practices
- **ALWAYS MERGE TO MAIN AS YOU GO — without asking.** After every commit to the
  dev branch, push it and merge it into `main`, then push `main`. Never let work
  sit only on the branch. A parallel agent also commits here, so if `main` has
  diverged, reconcile with a non-fast-forward merge (`git merge <branch> --no-ff`)
  — never a force-push to `main`. Keep `main` current at all times.
- Develop on branch `claude/technosignature-research-t5hjpw`; commit and push.
  (This session's designated dev branch is `claude/lhs-1140b-signatures-search-gvxtj4`.)
- Acquisition (Gaia/VizieR/SPARCL/IRSA) is blocked in the sandbox but works on the
  GitHub runner; run searches via `.github/workflows/*.yml` (workflow_dispatch),
  which commit results back to the branch.
- Keep the contamination-rejection discipline: trace every candidate to a
  systematic before believing it; but the objective is a *detection*, not a limit.

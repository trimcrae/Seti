#!/usr/bin/env bash
#
# Rebuild METRONOME's summary.json / candidates.json from EVERY per-star record
# on the branch AS IT NOW STANDS, commit that, and repeat until the committed
# summary accounts for every per-star vet file on the branch.
#
#   RESULTS_BRANCH=<branch> scripts/metronome_reconcile_landed.sh
#
# WHY.  MEASURED 2026-09-23: five `vetstar` runs vetted five stars in parallel.
# Each reconciled summary.json/candidates.json in its OWN checkout, and
# commit_results.sh is last-writer-wins, so the landed files carried one
# demotion of five.  Per-star vet files (vetstar_<mission>_<id>.json) make the
# RECORDS collision-free; this step makes the SUMMARY follow them: it runs
# after this run's own record has landed, reads the branch head (which holds
# every record that landed before it), and rebuilds.  A run whose record lands
# later does the same after it, so the last rebuild to land has seen every
# record -- and the loop re-checks that on the remote rather than assuming it.
set -uo pipefail
BRANCH="${RESULTS_BRANCH:-main}"
DIR=results/metronome

for attempt in 1 2 3 4; do
  git fetch --quiet origin "$BRANCH" || { echo "reconcile_landed: fetch failed"; sleep 5; continue; }
  # the branch head's records, over this run's working copy
  git checkout "origin/$BRANCH" -- "$DIR" 2>/dev/null || true
  python -m seti.metronome.run --stage reconcile || { echo "::error::reconcile failed"; exit 1; }
  scripts/commit_results.sh "metronome: reconcile summary from every per-star record" \
    "$DIR/summary.json" "$DIR/candidates.json" || exit 1
  git fetch --quiet origin "$BRANCH" || true
  n_files=$(git ls-tree --name-only "origin/$BRANCH" "$DIR/" \
            | grep -E "^$DIR/vetstar_[^/]+\.json$" | grep -vc "vetstar_fold_" || true)
  n_sum=$(git show "origin/$BRANCH:$DIR/summary.json" 2>/dev/null \
          | python -c "import json,sys; print((json.load(sys.stdin).get('vetstar') or {}).get('n_vetted', -1))" \
          2>/dev/null || echo -1)
  echo "reconcile_landed: attempt $attempt -- $n_files per-star vet files on $BRANCH, summary accounts for $n_sum"
  if [ "$n_files" = "$n_sum" ]; then
    exit 0
  fi
  sleep $((attempt * 7))
done
echo "::warning::reconcile_landed: summary on $BRANCH still does not account for every vet file"
exit 0

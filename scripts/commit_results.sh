#!/usr/bin/env bash
#
# Commit generated result files to a branch, or FAIL LOUDLY.
#
#   scripts/commit_results.sh "commit message" path [path...]
#
# WHY THIS EXISTS.  On 2026-08-27 `tocsin-altfeeds` run 33029076779 spent 169
# minutes walking the ATLAS queue, wrote its ledger, committed it -- and landed
# nothing on `main`, in a run that went GREEN.  The step was the pattern every
# workflow here used:
#
#     git commit -m "..."
#     git pull --rebase --autostash origin main || true
#     git push origin HEAD:main || { sleep 5; git pull --rebase ...; git push ...; }
#
# `main` had moved during those 169 minutes, the rebase hit a conflict in
# census.json, and `|| true` swallowed it -- leaving a REBASE IN PROGRESS with
# the commit unapplied.  `git push` then had nothing to push, said "Everything
# up-to-date", and exited 0.  The retry could not help: the second
# `git pull --rebase` fails while a rebase is in progress, and its `|| true`
# swallowed that too.  Green run, no result, no alert -- and because the file it
# would have refreshed simply did not change, the staleness check in alerts.py
# reads the channel as merely a little old rather than broken.
#
# TWO RULES, and the second is the one that makes this trustworthy:
#
#   1. NEVER REBASE.  These are generated artefacts, wholly rewritten by the run
#      that produces them; there is no line-level history to preserve and a
#      three-way merge of two machine-written JSON files is meaningless even
#      when it succeeds.  So the working copy is re-created on top of whatever
#      the branch head is NOW and the files are laid down over it.  A conflict
#      is then not possible -- last-writer-wins, stated rather than discovered.
#
#   2. VERIFY THE RESULT, NOT THE CALL.  A zero exit from `git push` is not
#      evidence that anything was pushed; "Everything up-to-date" is also zero.
#      So after pushing, the commit must be an ancestor of the REMOTE ref, read
#      back with `git ls-remote`.  That is the check the old pattern lacked, and
#      it is the only one that could have caught this.
#
# Anything unresolvable exits non-zero, which turns the run red -- so the hourly
# failure sweep sees it.  A silent drop is worse than a loud failure: the run
# that drops silently is indistinguishable from one that had nothing to say.
set -uo pipefail

MSG="${1:-}"; shift || true
if [ -z "$MSG" ] || [ "$#" -eq 0 ]; then
  echo "usage: commit_results.sh MESSAGE PATH [PATH...] [--prune PATH...]" >&2
  exit 2
fi

BRANCH="${RESULTS_BRANCH:-main}"
ATTEMPTS="${RESULTS_PUSH_ATTEMPTS:-5}"

# ---------------------------------------------------------------------------
# Take the generated files aside FIRST.  Everything below re-creates the working
# tree from the remote, and files that only exist in this run must survive that.
# ---------------------------------------------------------------------------
# `--prune PATH` marks a path whose ABSENCE is meaningful and must be committed
# as a deletion.  loom-catalogue's `catalogue.inprogress.json` is the case: a run
# that FINISHED deletes its checkpoint, and a stale checkpoint left in git is a
# resume point offered for a run that already ended.  Ordinary paths are only
# ever added; a run that did not produce one is not asserting it should go.
stage="$(mktemp -d)"
present=()
prune=()
mode="keep"
for p in "$@"; do
  if [ "$p" = "--prune" ]; then mode="prune"; continue; fi
  if [ -e "$p" ]; then
    mkdir -p "$stage/$(dirname "$p")"
    cp -R "$p" "$stage/$(dirname "$p")/"
    present+=("$p")
  elif [ "$mode" = "prune" ]; then
    prune+=("$p")
    echo "commit_results: $p absent; its deletion will be committed"
  else
    echo "commit_results: $p not produced by this run; skipping it"
  fi
done
if [ "${#present[@]}" -eq 0 ] && [ "${#prune[@]}" -eq 0 ]; then
  echo "commit_results: this run produced none of the requested paths; nothing to commit"
  exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
# A rebase or merge left over from an earlier step would poison every command
# below, and leaving one behind is the failure this script exists to end.
git rebase --abort >/dev/null 2>&1 || true
git merge --abort  >/dev/null 2>&1 || true

for attempt in $(seq 1 "$ATTEMPTS"); do
  if ! git fetch --quiet origin "$BRANCH"; then
    echo "commit_results: fetch of $BRANCH failed (attempt $attempt/$ATTEMPTS)"
    sleep $((attempt * 3)); continue
  fi
  # Rule 1: re-create the tree at the CURRENT head, then lay our files over it.
  # `reset --hard` does not touch untracked files, so anything else this run
  # generated (and did not ask to commit) is still there for later steps.
  git reset --hard --quiet "origin/$BRANCH" || { echo "commit_results: reset failed" >&2; exit 1; }
  for p in ${present[@]+"${present[@]}"}; do
    mkdir -p "$(dirname "$p")"
    rm -rf "$p"
    cp -R "$stage/$p" "$p"
  done
  # `reset --hard` brought a pruned path back from the branch; this run says it
  # should not be there, so remove it and stage that with `-A`.
  for p in ${prune[@]+"${prune[@]}"}; do
    rm -rf "$p"
  done

  if [ "${#present[@]}" -gt 0 ]; then
    git add -f -- "${present[@]}" || { echo "commit_results: add failed" >&2; exit 1; }
  fi
  for p in ${prune[@]+"${prune[@]}"}; do
    git add -A -f -- "$p" || { echo "commit_results: add -A failed" >&2; exit 1; }
  done
  if git diff --cached --quiet; then
    echo "commit_results: results are identical to $BRANCH; nothing to commit"
    exit 0
  fi
  git commit --quiet -m "$MSG" || { echo "commit_results: commit failed" >&2; exit 1; }
  sha="$(git rev-parse HEAD)"

  if git push --quiet origin "HEAD:$BRANCH"; then
    # Rule 2: a zero exit is not evidence.  Ask the REMOTE what it has.
    remote="$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')"
    if [ "$remote" = "$sha" ] || git merge-base --is-ancestor "$sha" "$remote" 2>/dev/null; then
      echo "commit_results: ${sha:0:7} is on $BRANCH (remote now ${remote:0:7})"
      exit 0
    fi
    echo "commit_results: push reported success but $BRANCH is at ${remote:0:7}," \
         "which does not contain ${sha:0:7} -- retrying"
  else
    echo "commit_results: push rejected (attempt $attempt/$ATTEMPTS); refetching"
  fi
  sleep $((attempt * 3))
done

echo "commit_results: FAILED to land ${present[*]-} on $BRANCH after $ATTEMPTS attempts." >&2
echo "commit_results: the results are NOT lost -- they are in this run's artifact." >&2
exit 1

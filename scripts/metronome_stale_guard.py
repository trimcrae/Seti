#!/usr/bin/env python
"""Refuse to commit results produced by code OLDER than what the branch holds.

`scripts/commit_results.sh` is deliberately last-writer-wins: it never rebases,
it lays the files down over whatever the branch head is now, and a conflict is
therefore impossible.  That is the right rule for a queue that runs in order.
It is the wrong rule for a queue that does not.

MEASURED, 2026-09-22.  Metronome run 35741300225 was dispatched at 14:34 UTC
from commit 22e7b3d2, sat behind the account's Actions ceiling for nine hours
while two newer runs overtook it, and at 23:37 UTC committed its assess output
over run 35746944111's.  Its code predated `population_period`, `few_cycles`
and the catalogue-epoch stack, so `summary.json` went from 3,131 scanned /
53 watch / 5 interest / 1 candidate with nine light-curve demotions to
3,040 / 19 / 0 / 1 with no `redetect` block at all — a silent regression in a
run that was green, which is exactly the failure mode `commit_results.sh` was
written to prevent in a different guise.

**The artefact's own `generated_utc` cannot detect this**, and it is worth
saying why, because it is the obvious thing to reach for and it points the
wrong way: the stale run WROTE its file last, so its timestamp is the newer
one.  Wall-clock recency is precisely the property a queue-delayed run has.

What is actually stale is the CODE.  So the test is:

  1. find the commit that last wrote this result file on the branch (C);
  2. if this run's checkout (S) is an ancestor of C and S != C, then C was
     produced by code that already contains everything S has, and more;
  3. unless the channel's own sources are identical between S and C — a plain
     re-run of the same code is not stale — in which case allow it.

Ancestry needs history, and `actions/checkout` is shallow by default, so the
branch is deepened first.  Every step fails OPEN: if the history is not there,
or the file has no recorded writer, or git cannot answer, the run commits.  A
guard that fired on ignorance would block every run whose artefact or layout
changed, which is worse than the failure it prevents.

Usage:

    python scripts/metronome_stale_guard.py <branch> <file> [<file> ...]

Writes ``stale=true|false`` to ``$GITHUB_OUTPUT`` and always exits 0 — the
caller gates its commit step on the flag, so a stale run finishes green with
its artefacts uploaded and simply does not overwrite the branch.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Changing any of these changes what a result file MEANS, so two runs that
#: differ here are not interchangeable.  Everything else (docs, other
#: channels, STATUS) may differ freely between two runs of the same code.
CHANNEL_SOURCES = ("src/seti/metronome", "config/metronome.yaml",
                   ".github/workflows/metronome.yml")


def _git(*args, check: bool = False) -> str | None:
    try:
        r = subprocess.run(["git", *args], check=check, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _deepen(branch: str) -> None:
    for extra in (["--deepen=500"], ["--unshallow"], []):
        if _git("fetch", "--quiet", *extra, "origin", branch) is not None:
            return


def _last_writer(branch: str, path: str) -> str | None:
    for ref in (f"origin/{branch}", branch, "HEAD"):
        out = _git("log", "-1", "--format=%H", ref, "--", path)
        if out:
            return out
    return None


def _is_ancestor(a: str, b: str) -> bool | None:
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                           capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None                                   # 128: missing object, unknown


def _sources_differ(a: str, b: str) -> bool | None:
    try:
        r = subprocess.run(["git", "diff", "--quiet", a, b, "--", *CHANNEL_SOURCES],
                           capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode == 0:
        return False
    if r.returncode == 1:
        return True
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: metronome_stale_guard.py <branch> <file> [<file> ...]")
        return 0
    branch, files = argv[0], argv[1:]
    mine = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    if not mine:
        print("[guard] cannot read this run's commit — failing open")
        return _emit(False)
    _deepen(branch)

    stale = False
    for f in files:
        writer = _last_writer(branch, f)
        if not writer:
            print(f"[guard] {f}: no recorded writer on {branch} — ok")
            continue
        if writer == mine:
            print(f"[guard] {f}: written by this very commit — ok")
            continue
        anc = _is_ancestor(mine, writer)
        if anc is None:
            print(f"[guard] {f}: ancestry unavailable ({mine[:8]} vs {writer[:8]}) — ok")
            continue
        if not anc:
            print(f"[guard] {f}: this run ({mine[:8]}) is not behind its writer "
                  f"({writer[:8]}) — ok")
            continue
        differ = _sources_differ(mine, writer)
        if differ is False:
            print(f"[guard] {f}: writer {writer[:8]} is ahead of {mine[:8]} but the "
                  "channel's sources are identical — a re-run, ok")
            continue
        print(f"[guard] {f}: STALE — {writer[:8]} wrote it with code this run "
              f"({mine[:8]}) predates")
        stale = True

    if stale:
        print("::warning::this run's CODE is older than the code that produced the "
              "results now on the branch; not committing.  A run that queued for hours "
              "runs the commit it was dispatched from, and commit_results.sh is "
              "last-writer-wins.  Its artifacts are still uploaded.")
    return _emit(stale)


def _emit(stale: bool) -> int:
    out = os.environ.get("GITHUB_OUTPUT")
    flag = "true" if stale else "false"
    if out:
        with open(out, "a") as fh:
            fh.write(f"stale={flag}\n")
    print(f"stale={flag}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())

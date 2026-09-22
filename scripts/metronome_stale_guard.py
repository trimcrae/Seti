#!/usr/bin/env python
"""Refuse to commit results that are OLDER than the ones already on the branch.

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

A checkout-commit ancestry test cannot catch this: the stale run's commit IS
an ancestor of the branch head.  What distinguishes it is the artefact's own
clock.  So each result file carries `generated_utc`, and this reads the copy
currently on the branch and compares.  If the branch already holds a NEWER
version of any file this run is about to write, the run is stale and must not
commit.

Usage:

    python scripts/metronome_stale_guard.py <branch> <file> [<file> ...]

Writes ``stale=true|false`` to ``$GITHUB_OUTPUT`` and always exits 0 — the
caller gates its commit step on the flag, so a stale run finishes green with
its artefacts uploaded and simply does not overwrite the branch.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse(ts) -> _dt.datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            d = _dt.datetime.strptime(ts.strip(), fmt)
        except ValueError:
            continue
        return d if d.tzinfo else d.replace(tzinfo=_dt.UTC)
    return None


def _generated(blob: str) -> _dt.datetime | None:
    try:
        d = json.loads(blob)
    except (ValueError, TypeError):
        return None
    return _parse(d.get("generated_utc")) if isinstance(d, dict) else None


def _on_branch(branch: str, path: str) -> str | None:
    for ref in (f"origin/{branch}", branch):
        try:
            return subprocess.run(["git", "show", f"{ref}:{path}"], check=True,
                                  capture_output=True, text=True).stdout
        except (subprocess.CalledProcessError, OSError):
            continue
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: metronome_stale_guard.py <branch> <file> [<file> ...]")
        return 0
    branch, files = argv[0], argv[1:]
    try:
        subprocess.run(["git", "fetch", "--quiet", "origin", branch], check=False,
                       capture_output=True, text=True)
    except OSError:
        pass

    stale = False
    for f in files:
        local = Path(f)
        if not local.exists():
            print(f"[guard] {f}: not written by this run, skipped")
            continue
        mine = _generated(local.read_text(errors="replace"))
        blob = _on_branch(branch, f)
        theirs = _generated(blob) if blob is not None else None
        if mine is None or theirs is None:
            print(f"[guard] {f}: mine={mine} branch={theirs} — no comparison possible")
            continue
        verdict = "STALE" if theirs > mine else "ok"
        print(f"[guard] {f}: mine={mine.isoformat()} branch={theirs.isoformat()} — {verdict}")
        if theirs > mine:
            stale = True

    if stale:
        print("::warning::this run's results are OLDER than the branch's; not committing. "
              "A run that queued for hours commits at the code it was dispatched from, "
              "and commit_results.sh is last-writer-wins.")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"stale={'true' if stale else 'false'}\n")
    print(f"stale={'true' if stale else 'false'}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())

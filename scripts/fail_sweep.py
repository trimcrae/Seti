#!/usr/bin/env python3
"""Runner entry point for the failure sweep (see ``seti.failsweep``).

Imports ``seti.failsweep`` and ``seti.cronwatch`` only -- not ``seti.cli``,
which pulls in numpy and pandas -- because this runs hourly inside `watchdog`
and needs nothing but ``requests``.

Usage (from the repository root):

    GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/name \\
        python scripts/fail_sweep.py [--no-retry] [--window-hours 24]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seti.cronwatch import ActionsApi  # noqa: E402
from seti.failsweep import WINDOW_HOURS, sweep  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--window-hours", type=float, default=WINDOW_HOURS)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    api = ActionsApi(args.repo, token) if (args.repo and token) else None
    if api is None:
        print("[failsweep] no GITHUB_REPOSITORY/GITHUB_TOKEN -- nothing was "
              "read and nothing was retried")

    rep = sweep(args.root, api=api, retry=not args.no_retry,
                window_hours=args.window_hours, out_dir=args.out_dir)
    print(f"[failsweep] considered={rep['n_considered']} "
          f"retried={rep.get('retried', 0)} skipped={rep['n_skipped']}")
    for rec in rep["retry"]:
        mark = "RETRIED" if rec.get("retry_dispatched") else "NOT retried"
        line = f"  {mark:<12} {rec['name']:<22} run {rec['id']} ({rec['conclusion']})"
        if rec.get("retry_error"):
            line += f"  -> {rec['retry_error']}"
        print(line)
    # Every refusal is printed.  A sweep that retries nothing because it could
    # not read a branch head looks identical to a healthy one in a bare count.
    for rec in rep["skipped"]:
        print(f"  {'skip:' + rec['reason']:<26} {rec['name']:<22} "
              f"run {rec['id']} ({rec['conclusion']})")
        if rec.get("note"):
            print(f"      {rec['note']}")
    unknown = sum(1 for r in rep["skipped"] if r["reason"] == "branch_head_unknown")
    if unknown:
        print(f"[failsweep] WARNING: {unknown} run(s) skipped because their "
              f"branch head could not be read -- the sweep is degraded, not clean")
    # Green even when it found failures: a red watchdog would bury the ledger
    # it just wrote behind a red run of its own.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Runner entry point for the scheduler watch (see ``seti.cronwatch``).

WHY THIS EXISTS ALONGSIDE ``seti.cli cron-watch``.  This runs inside the hourly
``watchdog`` job, and ``seti.cli`` imports numpy and pandas at module scope --
which means installing the whole scientific stack, every hour, to make two HTTP
calls per workflow.  This imports ``seti.cronwatch`` and nothing else, so the
job needs only ``pyyaml`` and ``requests``.  The behaviour is identical; the CLI
subcommand stays for anyone running it by hand in the full environment.

Usage (from the repository root):

    GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/name \\
        PYTHONPATH=src python scripts/cron_watch.py [--no-dispatch] [--ref main]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seti.cronwatch import SELF_HEAL_ONLY, ActionsApi, sweep  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--ref", default="main",
                    help="ref to dispatch catch-up runs on; scheduled runs only "
                         "ever fire on the default branch, so this is main")
    ap.add_argument("--no-dispatch", action="store_true")
    ap.add_argument("--self-heal-only", action="store_true",
                    help="dispatch ONLY the workflows in "
                         "seti.cronwatch.SELF_HEAL_ONLY -- the ones the single-"
                         "actor rule cannot cover because they are the actor")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    api = ActionsApi(args.repo, token) if (args.repo and token) else None
    if api is None:
        # Said out loud: a sweep with no API answers UNKNOWN for every workflow,
        # which is the honest reading and looks like a clean bill of health.
        print("[cronwatch] no GITHUB_REPOSITORY/GITHUB_TOKEN -- every workflow "
              "will be UNKNOWN and nothing will be dispatched")

    rep = sweep(args.root, api=api, ref=args.ref, dispatch=not args.no_dispatch,
                dispatch_only=SELF_HEAL_ONLY if args.self_heal_only else None,
                out_dir=args.out_dir)
    print(f"[cronwatch] workflows={rep['n_workflows']} "
          f"overdue={rep['n_overdue']} unknown={rep['n_unknown']} "
          f"dispatched={rep['n_dispatched']}")
    for wf in rep["workflows"]:
        if wf["status"] in ("OK", "WITHIN_GRACE"):
            continue
        line = (f"  {wf['status']:<10} {wf['workflow']:<28} "
                f"expected {wf['expected_last_fire_utc']} "
                f"last {wf['last_scheduled_run_utc'] or 'never'}")
        if wf.get("catchup_dispatched_utc"):
            line += "  -> CATCH-UP DISPATCHED"
        elif wf.get("catchup_error"):
            line += f"  -> catch-up FAILED: {wf['catchup_error']}"
        print(line)
    # Always green: a dropped firing is reported and re-fired, and failing the
    # watchdog on it would hide the failure sweep's own findings behind it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

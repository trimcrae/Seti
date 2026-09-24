"""RING brown-dwarf leg: one target list per run, shared by every NEOWISE shard.

WHY.  The bd acquire is a matrix of NEOWISE shards, each taking every n-th
target.  Before this, shard 0 fetched the target list fresh while shards > 0
read ``results/ring/bd/targets.csv`` -- which in a runner job is the COMMITTED
list of an earlier run, checked out from the branch.  The shards of one run
could therefore partition two different lists, so some targets were never
acquired and others twice.  (The RING channel's own fix makes every shard fetch
its own list; that removes the stale file, but N independent archive queries
are still N chances to disagree.)

So the workflow's ``bd-targets`` job fetches the list ONCE (``targets``) and
uploads it; every shard downloads that artifact and acquires through
``acquire``, which hands the downloaded list to ``stage_acquire`` via its
``fetchers["bd_targets"]`` hook.  Nothing in ``seti.ring`` changes.

    python scripts/ring_bd_handoff.py targets [--out DIR]
    python scripts/ring_bd_handoff.py acquire --shard i/n --targets DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

TARGETS = "targets.csv"
META = "acquire_targets.json"


def fetch_targets(cfg: dict, out_dir: Path, *, fetch=None) -> dict:
    """Fetch the bd target list once and write targets.csv + acquire_targets.json."""
    from seti.ring import acquire as acq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets, meta = (fetch or acq.fetch_bd_targets)(cfg)
    meta = dict(meta or {})
    meta["n_targets"] = int(len(targets))
    meta["handoff"] = "fetched once by the bd-targets job; every shard reads this list"
    if len(targets):
        targets.to_csv(out_dir / TARGETS, index=False)
    (out_dir / META).write_text(json.dumps(meta, indent=2, default=str))
    return meta


def handed_off_fetcher(targets_dir: Path):
    """A ``fetchers["bd_targets"]`` that returns the handed-off list.

    An absent list is returned as an EMPTY frame, so the shard records
    NO_DATA_REACHED; it never falls back to a fresh (possibly different) fetch.
    """
    targets_dir = Path(targets_dir)

    def _fetch(_cfg):
        p, m = targets_dir / TARGETS, targets_dir / META
        meta = json.loads(m.read_text()) if m.exists() else {}
        if not p.exists():
            meta.setdefault("error", f"handed-off target list {p} is missing")
            return pd.DataFrame(), meta
        return pd.read_csv(p), meta

    return _fetch


def acquire_shard(cfg: dict, out: Path, shard: int, n_shards: int, targets_dir: Path,
                  *, fetchers: dict | None = None) -> dict:
    from seti.ring.run import stage_acquire

    f = dict(fetchers or {})
    f["bd_targets"] = handed_off_fetcher(targets_dir)
    return stage_acquire(cfg, Path(out), "bd", shard=shard, n_shards=n_shards, fetchers=f)


def main(argv=None) -> int:
    from seti.ring.run import load_ring_config, out_root

    p = argparse.ArgumentParser(prog="ring_bd_handoff")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--out", default=None, help="results root (default results/ring)")
    a = sub.add_parser("acquire")
    a.add_argument("--shard", required=True, help="i/n")
    a.add_argument("--targets", required=True, help="directory holding the handed-off list")
    a.add_argument("--out", default=None, help="results root (default results/ring)")
    args = p.parse_args(argv)
    cfg = load_ring_config()
    root = out_root(args.out)
    if args.cmd == "targets":
        meta = fetch_targets(cfg, root / "bd")
        print(f"[ring] bd targets: {meta['n_targets']}")
        return 0
    i, n = (int(x) for x in args.shard.split("/"))
    rep = acquire_shard(cfg, root, i, n, Path(args.targets))
    print(f"[ring] bd shard {i}/{n}: {json.dumps(rep, default=str)[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

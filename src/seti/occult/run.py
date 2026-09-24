"""OCCULT stage orchestration: ``python -m seti.occult.run <stage> [...]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("results/occult")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti occult")
    ap.add_argument("stage", choices=["probe"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.stage == "probe":
        from .probe import run_probe

        s = run_probe(out)
        print(json.dumps({k: v for k, v in s.items() if k != "records"}, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
